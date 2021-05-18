import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils, centernet_utils


class HOSHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, voxel_size, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size #delete if unnecessary
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        self.class_names_each_head = []
        self.class_id_mapping_each_head = []

        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.quadrant = target_assigner_cfg.QUADRANT
        self.num_max_objs = target_assigner_cfg.NUM_MAX_OBJS 
        self.feature_map_stride = target_assigner_cfg.FEATURE_MAP_STRIDE        
        
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)

        total_classes = sum([len(x) for x in self.class_names_each_head])
        assert total_classes == len(self.class_names), f'class_names_each_head={self.class_names_each_head}'
        self.box_coder = getattr(box_coder_utils, target_assigner_cfg.BOX_CODER)(
            **target_assigner_cfg.BOX_CODER_CONFIG
        )

        self.forward_ret_dict = {}
        self.build_losses()

    def assign_target_of_single_head(self, gt_boxes, feature_map_size, num_max_objs):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size:

        Returns:

        """

        heatmap = gt_boxes.new_zeros(feature_map_size[0], feature_map_size[1])
        hos_box_labels = gt_boxes.new_zeros((feature_map_size[0] * feature_map_size[1],  gt_boxes.shape[-1] - 1 + 1))
        hos_box_code = gt_boxes.new_zeros((feature_map_size[0] * feature_map_size[1],  gt_boxes.shape[-1] - 1 + 1))
        quadrant_labels = gt_boxes.new_zeros((feature_map_size[0] * feature_map_size[1],  4))

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        l, w, h = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        
        x_stride = (self.point_cloud_range[3] - self.point_cloud_range[0]) / feature_map_size[0]
        y_stride = (self.point_cloud_range[4] - self.point_cloud_range[1]) / feature_map_size[1]
        z_stride = self.point_cloud_range[5] - self.point_cloud_range[2]
        x_offset = x_stride / 2
        y_offset = y_stride / 2
        
        x_shifts = torch.arange(
            self.point_cloud_range[0] + x_offset, self.point_cloud_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
        ).cuda()
        y_shifts = torch.arange(
            self.point_cloud_range[1] + y_offset, self.point_cloud_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
        ).cuda()
        z_shifts = x_shifts.new_tensor(z_stride / 2)
        
        center = torch.cat((x[:, None], y[:, None], l[:, None], w[:, None]), dim=-1)
        mask = (l > 0) & ( w >0)
        mask &= (self.point_cloud_range[0] <= x)
        mask &= (x <= self.point_cloud_range[3])
        mask &= (self.point_cloud_range[1] <= y)
        mask &= (y <= self.point_cloud_range[4])
        
        for k in range(gt_boxes[mask].shape[0]):

            hotspots, mask, quadrants = centernet_utils.obtain_cls_heatmap(heatmap, center[k], x_shifts, y_shifts, num_max_objs)
            arg_mask = torch.gt(mask.view(-1), 0)
            if arg_mask.sum() == quadrants.shape[0]:
                quadrant_labels[arg_mask] = quadrants.cuda().float()
            if arg_mask.sum() > 0:    
                hos_box_code[arg_mask, 0:2] = hotspots.to(hos_box_code.device)
                hos_box_code[arg_mask, 2] = z_shifts
           
                # obtain reg labels for every gt
                hos_box_labels[arg_mask] = self.box_coder.encode_torch(gt_box = gt_boxes[k, :-1], hotspots = hos_box_code[arg_mask])
        
        return heatmap, hos_box_labels, quadrant_labels


    def build_losses(self):
        self.add_module('cls_loss_func', loss_utils.BinaryFocalClassificationLoss())
        self.reg_loss_func = nn.functional.smooth_l1_loss
        self.spa_loss_func = nn.functional.binary_cross_entropy


    def get_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_preds = self.forward_ret_dict['box_preds']
        spa_preds = self.forward_ret_dict['spa_preds']
        heatmaps = self.forward_ret_dict['heatmaps']
        hos_box_labels = self.forward_ret_dict['hos_box_labels']
        quadrant_labels = self.forward_ret_dict['quadrant_labels']
        
        code_size = self.box_coder.code_size
        quadrant_size = self.quadrant        
        batch_size = heatmaps.shape[0]
        
        
        tb_dict = {}
        loss = 0
        cls_loss = 0
        mask = 0
        # classification loss
        cls_preds = cls_preds.permute(3, 0, 1, 2).contiguous()
        heatmaps = heatmaps.permute(1, 0, 2, 3).contiguous()
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            targets = heatmaps[idx]
            positives = (targets > 0)
            negatives = (targets == 0)
            cls_weights = (positives + negatives).float()
            pos_normalizer = torch.sum(torch.sum(cls_weights, dim=1), dim=1)
            pos_norm = pos_normalizer.unsqueeze(-1).unsqueeze(-1).expand_as(cls_weights)
            cls_weights /= torch.clamp(pos_norm, min=1)
            
            hm_loss = self.cls_loss_func(cls_preds[idx], targets, cls_weights)
            tb_dict['hm_loss_head_%d' % idx] = (hm_loss.sum() / batch_size).item()
            
            cls_loss += hm_loss/batch_size
            mask += positives 
        
        # regression loss
        box_preds = box_preds.view(-1,code_size)
        hos_box_labels = hos_box_labels.view(-1,code_size) 
        reg_mask = torch.gt(mask.view(-1), 0)
        hots_preds = box_preds[reg_mask]
        hots_labels = hos_box_labels[reg_mask]
        reg_loss = self.reg_loss_func(hots_preds, hots_labels)
        ## reg_loss is divided by 8
        #reg_weights = box_preds.new_ones(box_preds.shape)
        #reg_normalizer = reg_mask.sum().float()
        #reg_weights /= torch.clamp(reg_normalizer, min=1)
        #hots_weights = reg_weights[reg_mask]
        #reg_loss = self.reg_loss_func(hots_preds, hots_labels, hots_weights)
        reg_loss = 8 * reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        tb_dict['loc_loss_head'] =(reg_loss.sum()).item()
          
        # spatial relationship loss
        spa_loss = 0
        quadrant_labels = quadrant_labels.view(-1,quadrant_size) 
        spa_preds = spa_preds.view(-1,quadrant_size)
        hots_quad_labels = quadrant_labels[reg_mask]
        hots_quad_preds = spa_preds[reg_mask]
        hots_quad_preds = hots_quad_labels * hots_quad_preds
        
        hots_quad_preds = torch.clamp(hots_quad_preds, min=0) 
        hots_quad_preds = torch.clamp(hots_quad_preds, max=1) 
        for idx in range(4):
            spa_loss_sub = self.spa_loss_func(hots_quad_preds[:,idx].squeeze(),hots_quad_labels[:,idx].squeeze())
            spa_loss += spa_loss_sub
        tb_dict['spa_loss_head'] =(spa_loss.sum()).item()
        
        loss = cls_loss.sum()  + reg_loss.sum() + spa_loss
        #loss = cls_loss.sum()  + reg_loss.sum()
        
        tb_dict['rpn_loss'] = loss
        return loss, tb_dict

    def generate_predicted_boxes(self, gt_boxes, batch_size, cls_preds, box_preds):
        """
        Args:
            batch_size: N
            cls_preds: (N, L, W, C)
            box_preds: (N, L*W, 8)

        Returns:

        """

        post_process_cfg = self.model_cfg.POST_PROCESSING
        
        # generate centers and cast feature_map to point_cloud field 
        feature_map_size = self.grid_size[:2] // self.feature_map_stride
        x_stride = (self.point_cloud_range[3] - self.point_cloud_range[0]) / feature_map_size[0]
        y_stride = (self.point_cloud_range[4] - self.point_cloud_range[1]) / feature_map_size[1]
        z_stride = self.point_cloud_range[5] - self.point_cloud_range[2]
        x_offset = x_stride / 2
        y_offset = y_stride / 2
        
        x_shifts = torch.arange(
            self.point_cloud_range[0] + x_offset, self.point_cloud_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
        ).cuda()
        y_shifts = torch.arange(
            self.point_cloud_range[1] + y_offset, self.point_cloud_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
        ).cuda()
        z_shifts = x_shifts.new_tensor(z_stride / 2)
        
        box_dict, cls_dict, sco_dict = [], [], [] 
        for idx in range(batch_size):
            batch_hm = cls_preds[idx].sigmoid_()
            batch_box = box_preds[idx]
            final_pred_dicts = centernet_utils.decode_bbox_from_heatmap(
                heatmap=batch_hm, boxes=batch_box, x_shifts=x_shifts, 
                y_shifts=y_shifts, z_shifts=z_shifts,
                score_thresh=post_process_cfg.SCORE_THRESH,
                K=post_process_cfg.MAX_OBJ_PER_SAMPLE,
                nms_thresh=post_process_cfg.NMS_CONFIG.NMS_THRESH
            )
            
            box_dict.append(final_pred_dicts['pred_boxes'])
            sco_dict.append(final_pred_dicts['pred_scores'])
            cls_dict.append(final_pred_dicts['pred_labels'])

        return box_dict, cls_dict, sco_dict

    def forward(self, **kwargs):
        raise NotImplementedError
