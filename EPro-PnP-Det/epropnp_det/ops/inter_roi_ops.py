"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import torch
import torch.nn.functional as F


def get_overlap_boxes(bboxes1, bboxes2):
    overlap_boxes = bboxes1.new_empty((bboxes1.size(0), bboxes2.size(0), 4))
    bboxes1 = bboxes1.unsqueeze(1)
    overlap_boxes[..., :2] = torch.max(bboxes1[..., :2], bboxes2[:, :2])
    overlap_boxes[..., 2:] = torch.min(bboxes1[..., 2:], bboxes2[:, 2:])
    pos_mask = (overlap_boxes[..., 2:] - overlap_boxes[..., :2]) > 0
    pos_mask = torch.all(pos_mask, dim=2)
    return pos_mask, overlap_boxes


def logsumexp_across_rois(roi_inputs, rois):
    """
    Args:
        roi_inputs (torch.Tensor): shape (bn, chn, rh, rw)
        rois (torch.Tensor): shape (bn, 5)
    Returns:
        Tensor, shape (bn, chn, rh, rw)
    """
    bn, kn, rh, rw = roi_inputs.size()
    # allocate memory, (bn, chn, rh, rw)
    rois_logsumexp = roi_inputs.clone()
    if bn > 0:
        roi_ids = rois[:, 0].round().int()
        roi_bboxes = rois[:, 1:]
        for roi_id in range(max(roi_ids) + 1):
            ids = (roi_id == roi_ids).nonzero(as_tuple=False).squeeze(-1)
            if len(ids) <= 1:
                continue
            boxes = roi_bboxes[ids]
            pos_mask, overlap_boxes = get_overlap_boxes(boxes, boxes)
            pos_mask.fill_diagonal_(False)
            for id_self, pos_mask_single, overlap_boxes_single \
                    in zip(ids, pos_mask, overlap_boxes):
                if not any(pos_mask_single):
                    continue
                ids_overlap = ids[pos_mask_single]
                num_overlap = ids_overlap.size(0)
                roi_input_self = roi_inputs[id_self]  # (chn, rh, rw)
                roi_inputs_overlap = roi_inputs[ids_overlap]
                bbox_self = roi_bboxes[id_self]
                bbox_overlap = roi_bboxes[ids_overlap]
                overlap_boxes_ = overlap_boxes_single[pos_mask_single]
                wh_bbox_self = bbox_self[2:] - bbox_self[:2]
                wh_bbox_overlap = bbox_overlap[:, 2:] - bbox_overlap[:, :2]
                # resample
                scale = wh_bbox_self / wh_bbox_overlap
                xy_tl_in_bbox_overlap = 2 * (
                        overlap_boxes_[:, :2] - bbox_overlap[:, :2]
                    ) / (bbox_overlap[:, 2:] - bbox_overlap[:, :2]) - 1
                xy_tl_in_bbox_self = 2 * (
                        overlap_boxes_[:, :2] - bbox_self[:2]
                    ) / (bbox_self[2:] - bbox_self[:2]) - 1
                affine_mat = wh_bbox_self.new_zeros((num_overlap, 2, 3))
                affine_mat[:, 0, 0] = scale[:, 0]
                affine_mat[:, 1, 1] = scale[:, 1]
                affine_mat[:, :, 2] = \
                    xy_tl_in_bbox_overlap - scale * xy_tl_in_bbox_self
                grid = F.affine_grid(affine_mat, (num_overlap, 1, rh, rw), align_corners=False)
                # (#ovlp, chn, rh, rw)
                roi_inputs_resample = F.grid_sample(
                    roi_inputs_overlap, grid, padding_mode='border', align_corners=False)
                # (#ovlp, rh, rw, 2) -> (#ovlp, rh, rw) -> (#ovlp, 1, rh, rw)
                valid_grid = torch.all((grid > -1) & (grid < 1), dim=3).unsqueeze(1)
                # allocate memory to avoid concat
                roi_inputs_resampled_cat_self = roi_inputs_resample.new_empty(
                    (num_overlap + 1, kn, rh, rw))
                # fill out-of-border with inf
                roi_inputs_resampled_cat_self[:-1] = roi_inputs_resample.masked_fill(
                    ~valid_grid, float('-inf'))
                roi_inputs_resampled_cat_self[-1] = roi_input_self
                # (chn, rh, rw)
                rois_logsumexp[id_self] = roi_inputs_resampled_cat_self.logsumexp(dim=0)
    return rois_logsumexp


def logsoftmax_across_rois(roi_inputs, rois, extra_dim=None):
    """
    Args:
        roi_inputs (torch.Tensor): shape (bn, chn, rh, rw)
        rois (torch.Tensor): shape (bn, 5)
    Returns:
        Tensor, shape (bn, chn, rh, rw)
    """
    if extra_dim:
        return roi_inputs - logsumexp_across_rois(roi_inputs, rois).logsumexp(
            dim=extra_dim, keepdim=True)
    else:
        return roi_inputs - logsumexp_across_rois(roi_inputs, rois)


def softmax_across_rois(roi_inputs, rois, extra_dim=None):
    """
    Args:
        roi_inputs (torch.Tensor): shape (bn, chn, rh, rw)
        rois (torch.Tensor): shape (bn, 5)
    Returns:
        Tensor, shape (bn, chn, rh, rw)
    """
    return logsoftmax_across_rois(roi_inputs, rois, extra_dim).exp()
