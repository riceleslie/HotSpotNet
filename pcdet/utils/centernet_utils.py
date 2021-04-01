import torch
import torch.nn.functional as F
import numpy as np
import array
import math
from itertools import product
from . import box_utils
import pdb
#import numba


def obtain_cls_heatmap(heatmap, center, x_shifts, y_shifts, num_max_objs):
    heatmap_mask = torch.zeros(heatmap.shape)
    mask = torch.zeros(heatmap.shape)
    x, y, l, w = center[0:4]   
    bottom = x + l / 2
    top = x - l / 2
    left = y - w / 2
    right = y + w / 2
    MAX = num_max_objs / l / w

    masks_x = ((x_shifts > top) & (x_shifts < bottom))
    masks_y = ((y_shifts > left) & (y_shifts < right))
    heatmap_mask[masks_x,:] = 1
    x_mask = (heatmap_mask>0)
    heatmap_mask[:,masks_y] += 1
    spots_mask = (heatmap_mask>1)
    num = spots_mask.sum()
    
    spots_x = x_shifts[masks_x]
    spots_y = y_shifts[masks_y]
    spots = torch.Tensor(list(product(spots_x, spots_y)))
    
    if num > MAX.cpu().long():
        spots_dist = torch.pow((spots[:,0]-x),2) + torch.pow((spots[:,1]-y),2)
        value, ind = spots_dist.sort()
        spots_dist[ind[0:num_max_objs]] = 1
        spots_dist[ind[num_max_objs:-1]] = -1
        spots_dist[ind[-1]] = -1
        hotspots = spots[torch.gt(spots_dist,0)]
        heatmap[spots_mask] = spots_dist.to(heatmap.device)
        mask[spots_mask] = spots_dist
    else:
        hotspots = spots
        heatmap[spots_mask] = 1
        mask[spots_mask] = 1
    
    if num > 0: 
        quadrant_1 = (hotspots[:,0] < x.cpu()) &(hotspots[:,1] < y.cpu())
        quadrant_2 = (hotspots[:,0] < x.cpu()) &(hotspots[:,1] > y.cpu())
        quadrant_3 = (hotspots[:,0] > x.cpu()) &(hotspots[:,1] > y.cpu())
        quadrant_4 = (hotspots[:,0] > x.cpu()) &(hotspots[:,1] < y.cpu())
        quadrant = torch.stack([quadrant_1, quadrant_2, quadrant_3, quadrant_4])
        quadrant = quadrant.permute(1,0)
    else:
        quadrant = torch.zeros(num_max_objs,4)
    
    return hotspots, mask, quadrant
            
def gaussian_radius(height, width, min_overlap=0.5):
    """
    Args:
        height: (N)
        width: (N)
        min_overlap:

    Returns:

    """
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = (b1 ** 2 - 4 * a1 * c1).sqrt()
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = (b2 ** 2 - 4 * a2 * c2).sqrt()
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = (b3 ** 2 - 4 * a3 * c3).sqrt()
    r3 = (b3 + sq3) / 2
    ret = torch.min(torch.min(r1, r2), r3)
    return ret


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian_to_heatmap(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom, radius - left:radius + right]
    ).to(heatmap.device).float()

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def boxes_iou(box_a, box_b):
    """
    Args:
        box_a: (xmin, xmax, ymin, ymax)
        box_b: (xmin, xmax, ymin, ymax)

    Returns:
        iou between box_a and box_b
    """
    assert box_a.shape == box_b.shape
    x_min = torch.max(box_a[0], box_b[0])
    x_max = torch.min(box_a[1], box_b[1])
    y_min = torch.max(box_a[2], box_b[2])
    y_max = torch.min(box_a[3], box_b[3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (box_a[1] - box_a[0]) * (box_a[3] - box_a[2])
    area_b = (box_b[1] - box_b[0]) * (box_b[3] - box_b[2])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a + area_b - a_intersect_b, min=1e-6)
    return iou

#@numba.jit(nopython=True)
def nms_with_iou(dets, thresh):
    scores = dets[:, 4].cpu().numpy()
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate iou between i and j box
            dist = boxes_iou(dets[i,0:4], dets[j,0:4])
            # ovr = inter / areas[j]
            if dist > thresh:
                suppressed[j] = 1
    return keep

def _gather_feat(feat, ind, mask=None):
    if len(feat.shape) == 1:
        feat = feat.unsqueeze(-1)
    dim = feat.size(1)
    ind = ind.unsqueeze(1).expand(ind.size(0), dim)
    feat = feat.gather(0, ind)
    if feat.shape[-1] == 1:
        feat = feat.squeeze(-1)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(1, 2, 0).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _topk(scores, K=40):
    height, width, num_class = scores.size()

    topk_scores, topk_inds = torch.topk(scores.flatten(0, 1),K,dim=0)
    topk_inds = topk_inds % (height * width)
    topk_xs = (topk_inds // width).float()
    topk_ys = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(-1), K)
    topk_classes = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(-1, 1), topk_ind).view(K)
    topk_ys = _gather_feat(topk_ys.view(-1, 1), topk_ind).view(K)
    topk_xs = _gather_feat(topk_xs.view(-1, 1), topk_ind).view(K)

    return topk_score, topk_inds, topk_classes, topk_ys, topk_xs


def decode_bbox_from_heatmap(heatmap, boxes, point_cloud_range, feature_map_size,
                                score_thresh=0.3, K=100, nms_thresh=0.01):
    
    scores, inds, class_ids, ys, xs = _topk(heatmap, K=K)
    distance = boxes[:,0:2]
    center_z = boxes[:,2]
    sides = boxes[:,3:6]
    rot_cos = boxes[:,6]
    rot_sin = boxes[:,7]
    #pdb.set_trace()
    # obtain top K predicted box information
    distance = _gather_feat(distance, inds).view(K,2)
    center_z = _gather_feat(center_z, inds).view(K,1)
    sides = _gather_feat(sides, inds).view(K,3)
    rot_sin = _gather_feat(rot_sin, inds).view(K,1) 
    rot_cos = _gather_feat(rot_cos, inds).view(K,1) 
    #pdb.set_trace()
    # generate centers and cast feature_map to point_cloud field 
    x_stride = (point_cloud_range[3] - point_cloud_range[0]) / feature_map_size[0]
    y_stride = (point_cloud_range[4] - point_cloud_range[1]) / feature_map_size[1]
    z_stride = point_cloud_range[5] - point_cloud_range[2]
    x_offset = x_stride / 2
    y_offset = y_stride / 2
    
    x_shifts = torch.arange(
        point_cloud_range[0] + x_offset, point_cloud_range[3] + 1e-5, step=x_stride, dtype=torch.float32,
    ).cuda()
    y_shifts = torch.arange(
        point_cloud_range[1] + y_offset, point_cloud_range[4] + 1e-5, step=y_stride, dtype=torch.float32,
    ).cuda()
    z_shifts = x_shifts.new_tensor(z_stride / 2)
    
    # obtain top K nuerons' centers of point_cloud field    
    xs = _gather_feat(x_shifts, xs.long())
    ys = _gather_feat(y_shifts, ys.long())
    #pdb.set_trace()
    
    # obatain bounding box information
    angle = torch.atan2(rot_sin, rot_cos)
    center_x = xs + distance[:,0]
    center_y = ys + distance[:,1]
    center_z = center_z + z_shifts
    side_l = torch.exp(sides[:,0])
    side_w = torch.exp(sides[:,1])
    side_h = torch.exp(sides[:,2])
    boxes_list = [center_x, center_y, center_z.squeeze(-1), side_l, side_w, side_h]
    boxes_list.append(angle.squeeze(-1))
    boxes_list.append(class_ids.float())
    final_box_preds = torch.stack(boxes_list, dim=-1)
    final_scores = scores.view(K)

   # assert post_center_limit_range is not None
   # mask = (final_box_preds[..., :3] >= post_center_limit_range[:3]).all(2)
   # mask &= (final_box_preds[..., :3] <= post_center_limit_range[3:]).all(2)

    # remove box below the thresh
    if score_thresh is not None:
        mask = (final_scores > score_thresh)
    else:
        mask = torch.ones(K)
    cur_boxes = final_box_preds[mask] # cur_mask = mask[k]
    cur_scores = final_scores[mask]
    # gathering box information for NMS with iou
    xmin = cur_boxes[:,0] - cur_boxes[:,3] / 2
    xmax = cur_boxes[:,0] + cur_boxes[:,3] / 2
    ymin = cur_boxes[:,1] - cur_boxes[:,4] / 2
    ymax = cur_boxes[:,1] + cur_boxes[:,4] / 2
    boxes_candidate = torch.stack([xmin, xmax, ymin, ymax, cur_scores], dim=-1)
    keep = nms_with_iou(boxes_candidate, nms_thresh)
    boxes_ind = torch.tensor(keep).cuda().long()
    boxes_keep = _gather_feat(cur_boxes, boxes_ind)
    scores_keep = _gather_feat(cur_scores, boxes_ind)
    ret_pred_dicts = {
        'pred_boxes': [],
        'pred_scores': [],
        'pred_labels': []
    }
    ret_pred_dicts['pred_boxes']=boxes_keep
    ret_pred_dicts['pred_labels']=boxes_keep[:,-1]
    ret_pred_dicts['pred_scores']=scores_keep
    #boxes_keep = cur_boxes[keep] 
        
   # for k in range(batch_size):
   #     cur_mask = mask[k]
   #     cur_boxes = final_box_preds[k, cur_mask]
   #     cur_scores = final_scores[k, cur_mask]
   #     cur_labels = final_class_ids[k, cur_mask]

   #     if circle_nms:
   #         centers = cur_boxes[:, [0, 1]]
   #         boxes = torch.cat((centers, scores.view(-1, 1)), dim=1)
   #         keep = _circle_nms(boxes, min_radius=min_radius, post_max_size=nms_post_max_size)

   #         cur_boxes = cur_boxes[keep]
   #         cur_scores = cur_scores[keep]
   #         cur_labels = cur_labels[keep]

   #     ret_pred_dicts.append({
   #         'pred_boxes': cur_boxes,
   #         'pred_scores': cur_scores,
   #         'pred_labels': cur_labels
   #     })
    return ret_pred_dicts
