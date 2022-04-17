"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
"""

import warnings
import torch
import torch.nn.functional as F

from pytorch3d.utils import ico_sphere
from pytorch3d.structures.meshes import Meshes
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras

from .misc import yaw_to_rot_mat, box_mesh
from .builder import CENTER_TARGETS


@CENTER_TARGETS.register_module()
class VolumeCenter(object):
    def __init__(self,
                 base_mesh=dict(type='box'),
                 faces_per_pixel=16,
                 output_stride=4,
                 render_stride=4,
                 occlusion_factor=0.0,
                 get_bbox_2d=False,
                 min_box_size=4.0,
                 max_gpu_obj=-1):
        if base_mesh['type'] == 'ellipsoid':
            self.base_mesh = ico_sphere(base_mesh.get('ico_sphere_level', 4))
        elif base_mesh['type'] == 'box':
            self.base_mesh = box_mesh()
        self.rasterizer = MeshRasterizer()
        assert faces_per_pixel <= torch.iinfo(torch.int8).max
        self.faces_per_pixel = faces_per_pixel
        self.output_stride = output_stride
        self.render_stride = render_stride
        self.occlusion_factor = occlusion_factor
        self.rend_bbox_2d = get_bbox_2d
        self.min_box_size = min_box_size
        self.max_gpu_obj = max_gpu_obj

    def get_centers_2d(self, bboxes_2d, bboxes_3d, obj_img_inds, img_dense_x2d_small, img_dense_x2d_mask_small,
                       cam_intrinsic, max_shape):
        """
        Args:
            bboxes_2d (torch.Tensor): (num_obj, 4)
            bboxes_3d (torch.Tensor): (num_obj, 7)
            obj_img_inds (torch.Tensor): (num_obj, )
            img_dense_x2d_small (torch.Tensor): (num_img, 2, h_out, w_out)
            img_dense_x2d_mask_small (torch.Tensor): (num_img, 1, h_out, w_out)
            cam_intrinsic (torch.Tensor): (num_img, 3, 3)
            max_shape (torch.Tensor): (2, ) in format [h, w]

        Returns:
            list[torch.Tensor]: centers_2d, tensor shape (num_obj_per_img, 2)
        """
        num_obj = bboxes_3d.size(0)
        num_img = cam_intrinsic.size(0)
        pad_shape = torch.ceil(max_shape / self.output_stride) * self.output_stride
        device = bboxes_3d.device
        if num_obj == 0:
            centers_2d = bboxes_3d.new_empty((0, 2))
            centers_2d_list = [centers_2d] * num_img
            bboxes_2d = bboxes_3d.new_empty((0, 4))
            bboxes_2d_list = [bboxes_2d] * num_img
            valid_mask = bboxes_3d.new_empty((0, ), dtype=torch.bool)
            num_obj_per_img_list = [0] * num_img
            return centers_2d, bboxes_2d, centers_2d_list, bboxes_2d_list, valid_mask, num_obj_per_img_list
        verts = self.base_mesh.verts_list()[0].to(device) * 0.5  # (vn, 3)
        faces = self.base_mesh.faces_list()[0].to(device)  # (fn, 3)
        vn = verts.size(0)
        fn = faces.size(0)
        verts = verts[None].expand(num_obj, -1, -1)  # (num_obj, vn, 3)
        verts_oc = verts * bboxes_3d[:, None, :3]

        # =====transform to opencv camera space=====
        rot_mats = yaw_to_rot_mat(bboxes_3d[:, 6])  # (num_obj, 3, 3)
        # (num_obj, vn, 3) = ((num_obj, 3, 3) @ (num_obj, 3, vn)) -> (num_obj, vn, 3) + (num_obj, 1, 3)
        verts_cam = (rot_mats @ verts_oc.transpose(-1, -2)).transpose(-1, -2) + bboxes_3d[:, None, 3:6]

        # =====join meshes as scenes=====
        verts_list = []  # list of scenes
        faces_list = []
        num_obj_per_img_list = []
        img_has_object_list = []
        for i in range(num_img):
            verts_scene = verts_cam[obj_img_inds == i]  # (num_obj_per_img, vn, 3)
            num_obj_per_img = verts_scene.size(0)
            img_has_object = num_obj_per_img > 0
            if img_has_object:
                faces_scene = faces.unsqueeze(0).expand(num_obj_per_img, -1, -1)  # (num_obj_per_img, fn, 3)
                faces_scene = faces_scene + torch.arange(
                    0, num_obj_per_img * vn, step=vn, device=device)[:, None, None]  # reindex
                verts_list.append(verts_scene.reshape(num_obj_per_img * vn, 3))
                faces_list.append(faces_scene.reshape(num_obj_per_img * fn, 3))
            num_obj_per_img_list.append(num_obj_per_img)
            img_has_object_list.append(img_has_object)
        scene_meshes = Meshes(verts=verts_list, faces=faces_list)

        # =====select images with objects=====
        img_has_object_list = torch.tensor(img_has_object_list, dtype=torch.bool, device=device)
        img_dense_x2d_small = img_dense_x2d_small[img_has_object_list]
        img_dense_x2d_mask_small = img_dense_x2d_mask_small[img_has_object_list]
        cam_intrinsic = cam_intrinsic[img_has_object_list]
        obj_img_inds_new = (img_has_object_list.cumsum(dim=0) - 1)[obj_img_inds]

        # =====camera conversion=====
        f = cam_intrinsic[:, [0, 1], [0, 1]]  # (num_img, 2) [fx, fy]
        p = cam_intrinsic[:, :2, 2]  # (num_img, 2) [px, py]
        half_shape = pad_shape / 2  # (2, ) [h, w]
        denom = torch.min(half_shape)
        cameras = PerspectiveCameras(
            focal_length=-f / denom,
            principal_point=(half_shape[[1, 0]] - (p + 0.5)) / denom,
            device=device)

        # =====rasterize=====
        h_rend, w_rend = int(pad_shape[0]) // self.render_stride, int(pad_shape[1]) // self.render_stride
        raster_settings = RasterizationSettings(
            image_size=(h_rend, w_rend),
            blur_radius=0.0,
            faces_per_pixel=self.faces_per_pixel,
            z_clip_value=1e-2)
        frags = self.rasterizer(scene_meshes, cameras=cameras, raster_settings=raster_settings)
        pix_to_face = frags.pix_to_face  # (num_img, h_rend, w_rend, faces_per_pixel)
        zbuf = frags.zbuf  # (num_img, h_rend, w_rend, faces_per_pixel)

        # =====post proc=====
        if num_obj > self.max_gpu_obj > 0:
            bboxes_2d = bboxes_2d.cpu()
            zbuf = zbuf.cpu()
            pix_to_face = pix_to_face.cpu()
            img_dense_x2d_small = img_dense_x2d_small.cpu()
            img_dense_x2d_mask_small = img_dense_x2d_mask_small.cpu()
            pad_shape = pad_shape.cpu()
            obj_img_inds_new = obj_img_inds_new.cpu()
        centers_2d, bboxes_2d, valid_mask = self.post_proc(
            zbuf, pix_to_face, img_dense_x2d_small, img_dense_x2d_mask_small,
            pad_shape, num_obj, obj_img_inds_new, bboxes_2d, fn)
        centers_2d = centers_2d.to(device)
        bboxes_2d = bboxes_2d.to(device)
        valid_mask = valid_mask.to(device)

        # masking & to list
        centers_2d = centers_2d[valid_mask]
        bboxes_2d = bboxes_2d[valid_mask]
        obj_img_inds = obj_img_inds[valid_mask]
        num_obj_per_img_list = torch.count_nonzero(
            obj_img_inds == torch.arange(num_img, device=obj_img_inds.device)[:, None],
            dim=1).tolist()
        centers_2d_list = centers_2d.split(num_obj_per_img_list, dim=0)
        bboxes_2d_list = bboxes_2d.split(num_obj_per_img_list, dim=0)
        return centers_2d, bboxes_2d, centers_2d_list, bboxes_2d_list, valid_mask, num_obj_per_img_list

    def post_proc(self, zbuf, pix_to_face, img_dense_x2d_small, img_dense_x2d_mask_small,
                  pad_shape, num_obj, obj_img_inds, bboxes_2d, fn):
        num_img, h_rend, w_rend, _ = zbuf.size()
        device = zbuf.device

        # =====indexing=====
        bg_mask = pix_to_face == -1
        obj_id = pix_to_face // fn
        obj_id[bg_mask] = -1
        obj_mask = torch.eq(
            obj_id, torch.arange(num_obj, dtype=torch.int64, device=device)[:, None, None, None, None]
        ).any(dim=1)  # (num_obj, h_rend, w_rend, faces_per_pixel)
        obj_mask_neg = ~obj_mask
        obj_valid_mask = torch.count_nonzero(obj_mask, dim=-1) == 2  # (num_obj, h_rend, w_rend)

        z_inds = torch.arange(
            self.faces_per_pixel, device=device, dtype=torch.int8
        )[None, None, None].expand(num_obj, h_rend, w_rend, -1).clone()
        obj_near_far_inds = torch.empty((2, num_obj, h_rend, w_rend),
                                        device=device, dtype=torch.long)
        # (num_obj, h_rend, w_rend)
        torch.argmin(z_inds.masked_fill_(obj_mask_neg, self.faces_per_pixel),
                     dim=-1, out=obj_near_far_inds[0])
        torch.argmax(z_inds.masked_fill_(obj_mask_neg, -1),
                     dim=-1, out=obj_near_far_inds[1])
        # (h_rend, w_rend, 2, num_obj)
        obj_near_far_inds = obj_near_far_inds.permute(2, 3, 0, 1)
        obj_near_far_inds += obj_img_inds * self.faces_per_pixel
        # (h_rend, w_rend, 2 * num_obj) -> (num_obj, h_rend, w_rend, 2)
        obj_z_near_far = torch.gather(
            zbuf.permute(1, 2, 0, 3).reshape(h_rend, w_rend, num_img * self.faces_per_pixel),
            dim=-1,
            index=obj_near_far_inds.reshape(h_rend, w_rend, 2 * num_obj)
        ).reshape(h_rend, w_rend, 2, num_obj).permute(3, 0, 1, 2)
        obj_z_thickness = obj_z_near_far[..., 1] - obj_z_near_far[..., 0]  # (num_obj, h_rend, w_rend)
        obj_z_thickness *= obj_valid_mask
        assert (obj_z_thickness[obj_valid_mask] >= 0).all()

        # =====volumetric occlusion=====
        if self.occlusion_factor > 0:
            obj_near_inds = obj_near_far_inds[..., 0, :]  # (h_rend, w_rend, num_obj)
            # (h_rend, w_rend, num_img * faces_per_pixel)
            all_thickness = zbuf.new_zeros((h_rend, w_rend, num_img * self.faces_per_pixel))
            all_thickness.scatter_(
                dim=-1,
                index=obj_near_inds,
                src=obj_z_thickness.permute(1, 2, 0))  # (h_rend, w_rend, num_obj)
            all_thickness.reshape_(h_rend, w_rend, num_img, self.faces_per_pixel)
            all_thickness.cumsum_(dim=-1).roll_(1, -1)
            all_thickness[..., 0] = 0
            # (h_rend, w_rend, num_obj) -> (num_obj, num_img, h_rend)
            obj_occlusion_thickness = torch.gather(
                all_thickness.reshape(h_rend, w_rend, num_img * self.faces_per_pixel),
                dim=-1,
                index=obj_near_inds
            ).permute(2, 0, 1)
            obj_z_thickness *= torch.exp(-self.occlusion_factor * obj_occlusion_thickness)

        # =====resampling=====
        denom = pad_shape[[1, 0]] / 2
        grid = (img_dense_x2d_small.permute(0, 2, 3, 1) + 0.5) / denom - 1
        # (2 * num_obj, h_rend, w_rend)
        obj_z_thickness_mask = torch.cat([obj_z_thickness, obj_valid_mask], dim=0)
        obj_z_thickness_mask = F.grid_sample(
            # (num_img, 2 * num_obj, h_rend, w_rend)
            obj_z_thickness_mask[None].expand(num_img, -1, -1, -1),
            grid,  # (num_img, h_out, w_out, 2)
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)  # (num_img, 2 * num_obj, h_out, w_out)
        obj_z_thickness_mask *= img_dense_x2d_mask_small
        h_out, w_out = img_dense_x2d_small.shape[-2:]
        # (num_obj, 2, h_out, w_out)
        obj_z_thickness_mask = obj_z_thickness_mask.reshape(
            num_img, 2, num_obj, h_out, w_out
        )[obj_img_inds, :, torch.arange(num_obj, device=device)]

        # =====get weighted center & bbox2d=====
        h_arange = torch.arange(h_out, device=device, dtype=torch.float32).mul_(self.output_stride)
        w_arange = torch.arange(w_out, device=device, dtype=torch.float32).mul_(self.output_stride)
        y, x = torch.meshgrid(h_arange, w_arange)
        points = torch.stack((x, y), dim=-1) + self.output_stride / 2  # (h_out, w_out, 2)

        obj_z_thickness = obj_z_thickness_mask[:, 0, :, :, None]  # (num_obj, h_out, w_out, 1)
        weight_sum = obj_z_thickness.sum(dim=(-3, -2))  # (num_obj, 1)
        centers_2d = (obj_z_thickness * points).sum(dim=(-3, -2)) / weight_sum  # (num_obj, 2)
        valid_mask = (weight_sum >= 1e-6).squeeze(-1)

        if self.rend_bbox_2d:
            obj_mask = obj_z_thickness_mask[:, 1] >= 0.5  # (num_obj, h_out, w_out)
            obj_mask_w_neg = ~obj_mask.any(dim=1)  # (num_obj, w_out)
            obj_mask_h_neg = ~obj_mask.any(dim=2)  # (num_obj, h_out)
            x_ = w_arange[None].expand(num_obj, -1).clone()
            y_ = h_arange[None].expand(num_obj, -1).clone()
            x1 = x_.masked_fill_(obj_mask_w_neg, w_out * self.output_stride).min(dim=1)[0]
            x2 = x_.masked_fill_(obj_mask_w_neg, 0.0).max(dim=1)[0]
            y1 = y_.masked_fill_(obj_mask_h_neg, h_out * self.output_stride).min(dim=1)[0]
            y2 = y_.masked_fill_(obj_mask_h_neg, 0.0).max(dim=1)[0]
            bboxes_2d = torch.stack((x1, y1, x2 + self.output_stride, y2 + self.output_stride), dim=1)  # (num_obj, 4)
        valid_mask &= ((bboxes_2d[..., 2:] - bboxes_2d[..., :2]) >= self.min_box_size).all(dim=1)

        if not valid_mask.all():
            warnings.warn('Some annotated objects are disgarded. This can be triggered by small objects'
                          ' or small value of faces_per_pixel or objects outside the canvas')
        return centers_2d, bboxes_2d, valid_mask
