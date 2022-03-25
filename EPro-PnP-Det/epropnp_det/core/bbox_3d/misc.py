"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import math
import numpy as np
import torch
from pytorch3d.structures.meshes import Meshes

from epropnp_det.ops.iou3d.iou3d_utils import nms_gpu


def gen_unit_noc(num_pts, device=None):
    indices = torch.arange(0, num_pts, dtype=torch.float32, device=device) + 0.5
    phi = torch.arccos(1 - 2 * indices / num_pts)
    theta = math.pi * (1 + 5**0.5) * indices
    xyz = torch.stack(
        (torch.cos(theta) * torch.sin(phi),
         torch.sin(theta) * torch.sin(phi),
         torch.cos(phi)), dim=-1)
    return xyz


def project_to_image_r_mat(
        x3d, r_mat, t_vec, cam_intrinsic, img_shapes, z_min=0.5, allowed_border=200,
        return_z=False, return_clip_mask=False):
    """
    Args:
        x3d (torch.Tensor): shape (*, num_points, 3)
        r_mat (torch.Tensor): shape (*, 3, 3)
        t_vec (torch.Tensor): shape (*, 3) in format [x, y, z]
        cam_intrinsic (torch.Tensor): shape (*, 3, 3)
        img_shapes (torch.Tensor): shape (*, 2)

    Returns:
        Tensor: x2d_proj, shape (*, num_points, 2)
    """
    proj_r_mats = cam_intrinsic @ r_mat  # (*, 3, 3)
    proj_t_vecs = cam_intrinsic @ t_vec.unsqueeze(-1)  # (*, 3, 1)
    # (*, num_points, 3) = ((*, 3, 3) @ (*, 3, num_points) + (*, 3, 1)).T
    xyz_proj = (proj_r_mats @ x3d.transpose(-1, -2) + proj_t_vecs).transpose(-1, -2)
    z_proj = xyz_proj[..., 2:]  # (*, num_points, 1)
    if return_clip_mask:
        z_clip_mask = z_proj < z_min
    z_proj = z_proj.clamp(min=z_min)
    x2d_proj = xyz_proj[..., :2] / z_proj  # (*, num_points, 2)
    # clip to border
    x2d_min = -allowed_border - 0.5  # Number
    x2d_max = img_shapes[..., None, [1, 0]] + (allowed_border - 0.5)  # (*, 1, 2)
    if return_clip_mask:
        x2d_clip_mask = (x2d_proj < x2d_min) | (x2d_proj > x2d_max)
        clip_mask = z_clip_mask.squeeze(-1) | x2d_clip_mask.any(-1)  # (*, num_points)
    x2d_proj = torch.min(x2d_proj.clamp(min=x2d_min), x2d_max)
    if not return_z:
        if not return_clip_mask:
            return x2d_proj
        else:
            return x2d_proj, clip_mask
    else:
        if not return_clip_mask:
            return x2d_proj, z_proj
        else:
            return x2d_proj, z_proj, clip_mask


def project_to_image(
        x3d, pose, cam_intrinsic, img_shapes, z_min=0.5, allowed_border=200,
        return_z=False, return_clip_mask=False):
    """
    Args:
        x3d (torch.Tensor): shape (*, num_points, 3)
        pose (torch.Tensor): shape (*, 4) in format [x, y, z, yaw]
        cam_intrinsic (torch.Tensor): shape (*, 3, 3)
        img_shapes (torch.Tensor): shape (*, 2)

    Returns:
        Tensor: x2d_proj, shape (*, num_points, 2)
    """
    r_mat = yaw_to_rot_mat(pose[..., 3])
    t_vec = pose[..., :3]
    return project_to_image_r_mat(x3d, r_mat, t_vec, cam_intrinsic, img_shapes, z_min,
                                  allowed_border, return_z, return_clip_mask)


def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw: (*)

    Returns:
        rot_mats: (*, 3, 3)
    """
    if isinstance(yaw, torch.Tensor):
        pkg = torch
        device_kwarg = dict(device=yaw.device)
    else:
        pkg = np
        device_kwarg = dict()
    sin_yaw = pkg.sin(yaw)
    cos_yaw = pkg.cos(yaw)
    # [[ cos_yaw, 0, sin_yaw],
    #  [       0, 1,       0],
    #  [-sin_yaw, 0, cos_yaw]]
    rot_mats = pkg.zeros(yaw.shape + (3, 3), dtype=pkg.float32, **device_kwarg)
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats


def rot_mat_to_yaw(rot_mat):
    """
    Args:
        rot_mat: (*, 3, 3)

    Returns:
        yaw: (*)
    """
    if isinstance(rot_mat, torch.Tensor):
        atan2 = torch.atan2
    else:
        atan2 = np.arctan2
    yaw = atan2(rot_mat[..., 0, 2] - rot_mat[..., 2, 0], rot_mat[..., 0, 0] + rot_mat[..., 2, 2])
    return yaw


def box_mesh():
    return Meshes(
        verts=[torch.tensor([[-1, -1,  1],
                             [ 1, -1,  1],
                             [-1,  1,  1],
                             [ 1,  1,  1],
                             [-1, -1, -1],
                             [ 1, -1, -1],
                             [-1,  1, -1],
                             [ 1,  1, -1]], dtype=torch.float32)],
        faces=[torch.tensor([[0, 1, 2],
                             [1, 3, 2],
                             [2, 3, 7],
                             [2, 7, 6],
                             [1, 7, 3],
                             [1, 5, 7],
                             [6, 7, 4],
                             [7, 5, 4],
                             [0, 4, 1],
                             [1, 4, 5],
                             [2, 6, 4],
                             [0, 2, 4]], dtype=torch.int)])


def compute_box_3d(bbox_3d):
    """
    Args:
        bbox_3d: (*, 7)

    Returns:
        corners: (*, 8, 3)
        edge_corner_idx: (12, 2)
    """
    bs = bbox_3d.shape[:-1]
    rotation_matrix = yaw_to_rot_mat(bbox_3d[..., 6])  # (*bs, 3, 3)
    edge_corner_idx = np.array([[0, 1],
                         [1, 2],
                         [2, 3],
                         [3, 0],
                         [4, 5],
                         [5, 6],
                         [6, 7],
                         [7, 4],
                         [0, 4],
                         [1, 5],
                         [2, 6],
                         [3, 7]])
    corners = np.array([[ 0.5,  0.5,  0.5],
                        [ 0.5,  0.5, -0.5],
                        [-0.5,  0.5, -0.5],
                        [-0.5,  0.5,  0.5],
                        [ 0.5, -0.5,  0.5],
                        [ 0.5, -0.5, -0.5],
                        [-0.5, -0.5, -0.5],
                        [-0.5, -0.5,  0.5]], dtype=np.float32)
    if isinstance(bbox_3d, torch.Tensor):
        edge_corner_idx = torch.from_numpy(edge_corner_idx).to(device=bbox_3d.device)
        corners = torch.from_numpy(corners).to(device=bbox_3d.device)
    corners = corners * bbox_3d[..., None, :3]  # (*bs, 8, 3)
    corners = (rotation_matrix[..., None, :, :] @ corners[..., None]).reshape(*bs, 8, 3) \
              + bbox_3d[..., None, 3:6]
    return corners, edge_corner_idx


def edge_intersection(corners, edge_corner_idx, clip_axis, clip_val, op, edge_valid_mask=None):
    """
    Args:
        corners: (bs, 8, 3/2)
        edge_corner_idx: (12, 2)
        clip_val: (bs, )
        edge_valid_mask: (bs, 12)
    """
    if op == 'greater':
        op = torch.greater
    elif op == 'less':
        op = torch.less
    if edge_valid_mask is None:
        edge_valid_mask = corners.new_ones(
            (corners.size(0), edge_corner_idx.size(0)), dtype=torch.bool)
    corners_inside = op(corners[..., clip_axis], clip_val[:, None])  # (bs, 8)
    # compute z intersection
    edges_0_inside = corners_inside[:, edge_corner_idx[:, 0]]  # (bs, 12)
    edges_1_inside = corners_inside[:, edge_corner_idx[:, 1]]  # (bs, 12)
    edges_clipped = (edges_0_inside ^ edges_1_inside) & edge_valid_mask  # (bs, 12)
    edges_clipped_idx = edges_clipped.nonzero()  # (num_nonzero, 2) in [bs_ind, edge_ind]
    if edges_clipped_idx.shape[0] > 0:
        edge_corner_idx_to_clip = edge_corner_idx[edges_clipped_idx[:, 1], :]  # (num_nonzero, 2)
        edges_0 = corners[edges_clipped_idx[:, 0], edge_corner_idx_to_clip[:, 0], :]  # (num_nonzero, 3)
        edges_1 = corners[edges_clipped_idx[:, 0], edge_corner_idx_to_clip[:, 1], :]  # (num_nonzero, 3)
        axval0 = edges_0[:, clip_axis]  # (num_nonzero, )
        axval1 = edges_1[:, clip_axis]
        clip_val_ = clip_val[edges_clipped_idx[:, 0]]
        weight_0 = axval1 - clip_val_  # (num_nonzero, )
        weight_1 = clip_val_ - axval0
        intersection = (edges_0 * weight_0[:, None] + edges_1 * weight_1[:, None]
                        ) * (1 / (axval1 - axval0)).clamp(min=-1e6, max=1e6)[:, None]  # (num_nonzero, 3)
        clip_idx = torch.where(op(axval0, clip_val_),
                               edge_corner_idx_to_clip[:, 1],
                               edge_corner_idx_to_clip[:, 0])  # (num_nonzero, )
        corners[edges_clipped_idx[:, 0], clip_idx, :] = intersection  # replace clipped corners with intersection
        corners_inside[edges_clipped_idx[:, 0], clip_idx] = True
        edge_valid_mask &= corners_inside[:, edge_corner_idx[:, 0]] & corners_inside[:, edge_corner_idx[:, 1]]
    else:
        edge_valid_mask &= edges_0_inside & edges_1_inside
    return corners, corners_inside, edge_valid_mask


def bboxes_3d_to_2d(bbox_3d, cam_intrinsic, imsize, z_clip=0.1, min_size=4.0, clip=False):
    """
    Args:
        bbox_3d: (bs, 7)
        cam_intrinsic: (bs, 3, 3)
        imsize: (bs, 2) in [h, w]
    """
    assert bbox_3d.dim() == 2
    bs = bbox_3d.size(0)
    if bs > 0:
        # (bs, 8, 3), (12, 2)
        corners, edge_corner_idx = compute_box_3d(bbox_3d)
        corners, in_front, edge_valid_mask = edge_intersection(
            corners, edge_corner_idx, 2, corners.new_tensor([z_clip]).expand(bs), 'greater')
        pts_2d = corners @ cam_intrinsic.transpose(-1, -2)
        pts_2d = pts_2d[..., :2] / pts_2d[..., 2:].clamp(min=z_clip) + 0.5  # (bs, 8, 2)
        in_canvas = in_front
        if clip:
            pts_2d, in_canvas_x0, edge_valid_mask = edge_intersection(
                pts_2d, edge_corner_idx, 0, corners.new_tensor([0]).expand(bs), 'greater', edge_valid_mask)
            pts_2d, in_canvas_y0, edge_valid_mask = edge_intersection(
                pts_2d, edge_corner_idx, 1, corners.new_tensor([0]).expand(bs), 'greater', edge_valid_mask)
            pts_2d, in_canvas_x1, edge_valid_mask = edge_intersection(
                pts_2d, edge_corner_idx, 0, imsize[:, 1], 'less', edge_valid_mask)
            pts_2d, in_canvas_y1, edge_valid_mask = edge_intersection(
                pts_2d, edge_corner_idx, 1, imsize[:, 0], 'less', edge_valid_mask)
            in_canvas = in_canvas & in_canvas_x0 & in_canvas_x1 & in_canvas_y0 & in_canvas_y1  # (bs, 8)
        not_in_canvas = ~in_canvas
        pts_2d[not_in_canvas] = imsize[:, None, [1, 0]].expand(-1, 8, -1)[not_in_canvas]
        x0y0 = pts_2d.min(dim=1)[0].clamp(min=0)  # (bs, 2)
        pts_2d[not_in_canvas] = 0
        x1y1 = torch.minimum(pts_2d.max(dim=1)[0], imsize[:, [1, 0]])
        bbox = torch.cat((x0y0, x1y1), dim=1)  # (bs, 4)
        bbox_valid_mask = (x1y1 - x0y0).min(dim=1)[0] >= min_size  # (bs, )
    else:
        bbox = bbox_3d.new_empty((0, 4))
        bbox_valid_mask = bbox_3d.new_empty((0, ), dtype=torch.bool)
    return bbox, bbox_valid_mask


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2  # l in bbox_3d
    half_h = boxes_xywhr[:, 3] / 2  # w in bbox_3d
    # x in cam coord
    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    # z in cam coord, mirrored_direction
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def batched_bev_nms(bbox_3d, batch_inds, nms_thr=0.25):
    """
    Args:
        bbox_3d (Tensor): tensor shape (N, 8+),
            in format [l, h, w, x, y, z, ry, score, ind, *]
        batch_inds (Tensor): tensor shape (N, )
        nms_thr (float)

    Returns:
        Tuple:
            bbox_3d_out (Tensor)
            keep_inds (Tensor)
    """
    n = bbox_3d.size(0)
    if n > 1:
        boxes_for_nms = xywhr2xyxyr(
            bbox_3d[:, [3, 5, 0, 2, 6]])
        offset_unit = (boxes_for_nms[:, :4].max() - boxes_for_nms[:, :4].min()) * 2
        boxes_for_nms[:, :4] = boxes_for_nms[:, :4] + (offset_unit * batch_inds)[:, None]
        keep_inds = nms_gpu(
            boxes_for_nms, bbox_3d[:, 7], nms_thr)
    else:
        keep_inds = bbox_3d.new_zeros(0, dtype=torch.int64)
    bbox_3d_out = bbox_3d[keep_inds]
    return bbox_3d_out, keep_inds
