import numpy as np
import torch.nn as nn
import torch
import pdb
from .hos_head_template import HOSHeadTemplate

class HOSHead(HOSHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range, voxel_size,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            voxel_size=voxel_size, predict_boxes_when_training=predict_boxes_when_training
        )
        
        pi = 0.01
        self.conv_shared = nn.Conv2d(
            input_channels, input_channels,
            kernel_size=3,
            padding=1
        )
        self.conv_cls = nn.Sequential(
            nn.Conv2d(
            input_channels, self.num_class,
            kernel_size=3,
            bias=-np.log((1 - pi) / pi),
            padding=1
            ),
            nn.Sigmoid()
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.box_coder.code_size,
            kernel_size=1
        )
        self.conv_spa = nn.Sequential(
            nn.Conv2d(
            input_channels, self.quadrant,
            kernel_size=1,
            ),
            nn.Softmax(dim=1)
        )
        self.init_weights()
    
    def init_weights(self):
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)
        nn.init.normal_(self.conv_shared.weight, mean=0, std=0.001)

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        feature_map_size = self.grid_size[:2] // self.feature_map_stride

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'hos_box_labels': [],
            'quadrant_labels': [],
        }

        heatmaps, hos_box_labels, quadrant_labels  = [], [], []
        all_names = np.array(['bg', *self.class_names])
        for bs_idx in range(batch_size):
            cur_gt_boxes = gt_boxes[bs_idx]
            heatmap_list, hos_box_list, quadrant_list, = [], [], []
            for class_idx, cur_class_names in enumerate(self.class_names_each_head):
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                # gather all the gt_box belonging to the same class
                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, hos_box_label, quadrant_label  = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head,
                    feature_map_size=feature_map_size, num_max_objs=self.num_max_objs)

                heatmap_list.append(heatmap)
                hos_box_list.append(hos_box_label)
                quadrant_list.append(quadrant_label)
                np.save(f'./visual/gt_box_%s.npy' % class_idx, gt_boxes_single_head.cpu().numpy())
                np.save(f'./visual/heatmap_%s.npy' % class_idx, heatmap.cpu().numpy())
                
            heatmaps.append(torch.stack(heatmap_list, dim=0))
            hos_box_labels.append(torch.stack(hos_box_list, dim=0))
            quadrant_labels.append(torch.stack(quadrant_list, dim=0))
        
        pdb.set_trace()
        ret_dict['heatmaps'] = torch.stack(heatmaps, dim=0)
        ret_dict['hos_box_labels'] = torch.stack(hos_box_labels, dim=0)
        ret_dict['quadrant_labels'] = torch.stack(quadrant_labels, dim=0)
        return ret_dict
    
    def forward(self, data_dict):
        spatial_features_2d = data_dict['spatial_features_2d']
        shared_features_2d = self.conv_shared(spatial_features_2d)
        
        #cls_preds: [1, 3, 200, 176]
        #box_preds: [1, 8, 200, 176]
        cls_preds = self.conv_cls(shared_features_2d)
        box_preds = self.conv_box(shared_features_2d)
        spa_preds = self.conv_spa(shared_features_2d) 
        #cls_preds: [1, 176, 200, 3]
        #box_preds: [1, 176, 200, 8]
        cls_preds = cls_preds.permute(0, 3, 2, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 3, 2, 1).contiguous()  # [N, H, W, C]
        spa_preds = spa_preds.permute(0, 3, 2, 1).contiguous()  # [N, H, W, C]
        
        box_preds = box_preds.view(data_dict['batch_size'], -1, self.box_coder.code_size)
        spa_preds = spa_preds.view(data_dict['batch_size'], -1, self.quadrant)
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds
        self.forward_ret_dict['spa_preds'] = spa_preds

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            self.forward_ret_dict.update(targets_dict)
        
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        return data_dict
