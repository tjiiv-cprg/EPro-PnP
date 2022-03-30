"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
"""

import torch
import torch.nn.functional as F
import cv2


def draw_orient_density(pose_opt, pose_samples, pose_sample_logweights, size=512,
                        saturation=0.5, sphere_opacity=0.6, sample_kernel=(5, 5), intensity_scale=50):
    bs = pose_opt.size(0)
    axis_point = torch.eye(3, dtype=torch.float32, device=pose_opt.device)

    sample_weight = pose_sample_logweights.softmax(dim=0)[..., None].expand(-1, -1, 3)  # (num_samples, bs, 3)
    q = pose_samples[..., 3:]
    w, v = q.split([1, 3], dim=-1)
    v_cross_axis = torch.cross(
        *torch.broadcast_tensors(v[..., None, :].clone(), axis_point), dim=-1)  # (num_samples, bs, 3, 3)
    # (num_samples, bs, 1, 1, 3) @ (3, 3, 1) -> (num_samples, bs, 3, 1)
    v_t_axis = (v[..., None, None, :] @ axis_point[..., None]).squeeze(-1)
    # (num_samples, bs, 3, 3)
    axisrot = w.square()[..., None, :] * axis_point + 2 * w[..., None, :] * v_cross_axis \
              + 2 * v[..., None, :] * v_t_axis - v[..., None, :] @ v[..., :, None] * axis_point
    draw_size = pose_opt.new_tensor([size, size], dtype=torch.long)
    axis2d = axisrot[..., :2] * draw_size.mul(0.4) + draw_size.div(2).sub(0.5)  # (num_samples, bs, 3, 2)
    axis2d_inds = axis2d.round().long()
    axis2d_inds = axis2d_inds[..., 1] * size + axis2d_inds[..., 0]  # (num_samples, bs, 3)
    visibility_mask = axisrot[..., 2] <= 0  # (num_samples, bs, 3)
    canvas = pose_opt.new_zeros((bs, size, size, 3))
    front = canvas.reshape(bs, -1, 3).transpose(0, 1).clone().scatter_add_(
        dim=0, index=axis2d_inds, src=sample_weight * visibility_mask).transpose(0, 1).reshape(bs, size, size, 3)
    back = canvas.reshape(bs, -1, 3).transpose(0, 1).clone().scatter_add_(
        dim=0, index=axis2d_inds, src=sample_weight * (~visibility_mask)).transpose(0, 1).reshape(bs, size, size, 3)
    front = F.avg_pool2d(front.permute(0, 3, 1, 2), sample_kernel, 1,
                         (sample_kernel[0] // 2, sample_kernel[1] // 2)).permute(0, 2, 3, 1)
    back = F.avg_pool2d(back.permute(0, 3, 1, 2), sample_kernel, 1,
                        (sample_kernel[0] // 2, sample_kernel[1] // 2)).permute(0, 2, 3, 1)
    front *= intensity_scale * sample_kernel[0] * sample_kernel[1]
    back *= intensity_scale * sample_kernel[0] * sample_kernel[1]
    colors = torch.eye(3, dtype=torch.float32, device=pose_opt.device
                       ) * saturation + (1 - saturation) / 2  # (3, 3)
    front_layer = torch.pow(colors, front[..., None]).prod(-2)  # (bs, *draw_size, 3)
    back_layer = torch.pow(colors, back[..., None]).prod(-2)  # (bs, *draw_size, 3)
    wh_arange = torch.arange(size, device=pose_opt.device, dtype=torch.float32
                             ).sub(draw_size[0].div(2).sub(0.5)).div(draw_size[0].mul(0.4))
    s = wh_arange.square()
    circle_mask = (s + s[:, None]) <= 1.0  # (*draw_size)
    circle_layer = 1 - circle_mask.float() * 0.5
    # (bs, *draw_size, 3)
    draw = (back_layer * sphere_opacity + circle_layer[..., None] * (1 - sphere_opacity)).cpu().numpy()

    q = pose_opt[..., 3:]
    w, v = q.split([1, 3], dim=-1)
    v_cross_axis = torch.cross(
        *torch.broadcast_tensors(v[..., None, :].clone(), axis_point), dim=-1)  # (bs, 3, 3)
    v_t_axis = (v[..., None, None, :] @ axis_point[..., None]).squeeze(-1)
    axisrot = w.square()[..., None, :] * axis_point + 2 * w[..., None, :] * v_cross_axis \
              + 2 * v[..., None, :] * v_t_axis - v[..., None, :] @ v[..., :, None] * axis_point
    axis2d = axisrot[..., :2] * draw_size.mul(0.4) + draw_size.div(2).sub(0.5)  # (bs, 3, 2)
    axis2d_inds = (axis2d * 8).round().int().cpu().numpy()
    origin = (draw_size.div(2).sub(0.5) * 8).round().int().cpu().numpy()
    colors = ((1, 0, 0),
              (0, 1, 0),
              (0, 0, 1))
    for i in range(bs):
        for j in range(3):
            cv2.line(draw[i], tuple(origin), tuple(axis2d_inds[i, j]), colors[j],
                     thickness=3, shift=3)

    draw *= front_layer.cpu().numpy()

    return draw
