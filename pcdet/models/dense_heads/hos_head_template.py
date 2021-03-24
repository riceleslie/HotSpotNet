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

    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, num_max_objs=64
        ):
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
        
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

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
        center_int = center.int()
        num_max_hots = np.floor(num_max_objs / x_stride / y_stride)
        num_max_hots = num_max_hots.astype(np.int16) 
        for k in range(gt_boxes.shape[0]):
            if l[k] <= 0 or w[k] <= 0:
                continue

            #if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
             #   continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            
            hotspots, mask, quadrant = centernet_utils.obtain_cls_heatmap(heatmap, center[k], x_shifts, y_shifts, num_max_hots)
            pdb.set_trace()
            arg_mask = torch.gt(mask.view(-1), 0)
            if arg_mask.sum() == quadrant.shape[0]:
                quadrant_labels[arg_mask] = quadrant.cuda().float()
            
            if arg_mask.sum() > 0:    
                hos_box_code[arg_mask, 0:2] = hotspots.to(hos_box_code.device)
                hos_box_code[arg_mask, 2] = z_shifts
           
                # obtain reg labels for every gt
                hos_box_labels[arg_mask] = self.box_coder.encode_torch(gt_box = gt_boxes[k, :-1], hotspots = hos_box_code[arg_mask])
        
        #np.save('./visual/gt_box_01.npy', center.cpu().numpy())
        #np.save('./visual/heatmap_01.npy', heatmap.cpu().numpy())
        return heatmap, hos_box_labels, quadrant_labels


    def build_losses(self):
        self.add_module('cls_loss_func', loss_utils.BinaryFocalClassificationLoss())
        self.add_module('reg_loss_func', loss_utils.HOSSmoothL1Loss())
        self.add_module('spa_loss_func', loss_utils.HOSBCELoss())


    def get_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_preds = self.forward_ret_dict['box_preds']
        spa_preds = self.forward_ret_dict['spa_preds']
        heatmaps = self.forward_ret_dict['heatmaps']
        hos_box_labels = self.forward_ret_dict['hos_box_labels']
        quadrant_labels = self.forward_ret_dict['quadrant_labels']
        
        code_size = box_preds.shape[-1]
        batch_size = heatmaps.shape[1]
        quadrant_size = spa_preds.shape[-1]        
        cls_preds = cls_preds.permute(3, 0, 1, 2).contiguous()
        heatmaps = heatmaps.permute(1, 0, 2, 3).contiguous()
        hos_box_labels = hos_box_labels.permute(1, 0, 2, 3).contiguous().sum(dim=0).view(-1,code_size) 
        box_preds = box_preds.view(-1,code_size)
        quadrant_labels = quadrant_labels.permute(1, 0, 2, 3).contiguous().sum(dim=0).view(-1,quadrant_size) 
        spa_preds = spa_preds.view(-1,quadrant_size)
        
        tb_dict = {}
        loss = 0
        cls_loss = 0
        mask = 0
        # classification loss
        for idx, cur_class_names in enumerate(self.class_names_each_head):
            targets = heatmaps[idx]
            positives = (targets > 0)
            negatives = (targets == 0)
            cls_weights = (positives + negatives).float()
            pos_normalizer = cls_weights.sum().float()
            cls_weights /= torch.clamp(pos_normalizer, min=1)
            
            hm_loss = self.cls_loss_func(cls_preds[idx], targets, cls_weights)
            
            tb_dict['hm_loss_head_%d' % idx] = (hm_loss.sum() / batch_size).item()
            
            cls_loss += hm_loss
         
        # regression loss
        reg_mask = positives.view(-1)
        reg_weights = box_preds.new_ones(box_preds.shape)
        reg_normalizer = positives.sum().float()
        reg_weights /= torch.clamp(reg_normalizer, min=1)
        hots_weights = reg_weights[reg_mask]
        hots_preds = box_preds[reg_mask]
        hots_labels = hos_box_labels[reg_mask]
        
        reg_loss = self.reg_loss_func(hots_preds, hots_labels, hots_weights)
        reg_loss = reg_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
          
        # spatial relationship loss
        spa_loss = self.spa_loss_func(spa_preds[reg_mask], quadrant_labels[reg_mask])
        #tb_dict['loc_loss_head' % idx] =(reg_loss.sum() / batch_size).item()
           
        loss = cls_loss.sum()  + reg_loss.sum() + spa_loss
        
        tb_dict['rpn_loss'] = loss
        return loss, tb_dict


    def forward(self, **kwargs):
        raise NotImplementedError
