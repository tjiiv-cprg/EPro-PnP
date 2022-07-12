"""
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import math
import torch
import numpy as np
from utils.utils import AverageMeter
from utils.img import im_norm_255
import cv2
from progress.bar import Bar
import os
import utils.fancy_logger as logger
import time

from ops.rotation_conversions import matrix_to_quaternion
from ops.pnp.camera import PerspectiveCamera
from ops.pnp.cost_fun import AdaptiveHuberPnPCost
from ops.pnp.levenberg_marquardt import LMSolver, RSLMSolver
from ops.pnp.epropnp import EProPnP6DoF


def train(epoch, cfg, data_loader, model, obj_info, criterions, optimizer=None):
    model.train()
    preds = {}
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    Loss_mc = AverageMeter()
    Loss_t = AverageMeter()
    Loss_r = AverageMeter()
    Norm_factor = AverageMeter()
    Grad_norm = AverageMeter()
    num_iters = len(data_loader)
    bar = Bar('{}'.format(cfg.pytorch.exp_id[-60:]), max=num_iters)

    time_monitor = False
    vis_dir = os.path.join(cfg.pytorch.save_path, 'train_vis_{}'.format(epoch))
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    cam_intrinsic_np = cfg.dataset.camera_matrix.astype(np.float32)
    cam_intrinsic = torch.from_numpy(cam_intrinsic_np).cuda(cfg.pytorch.gpu)

    epropnp = EProPnP6DoF(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=5,
            init_solver=RSLMSolver(
                dof=6,
                num_points=16,
                num_proposals=4,
                num_iter=3))).cuda(cfg.pytorch.gpu)

    for i, (obj, obj_id, inp, target, loss_msk, trans_local, pose, c_box, s_box, box) in enumerate(data_loader):
        cur_iter = i + (epoch - 1) * num_iters
        if cfg.pytorch.gpu > -1:
            inp_var = inp.cuda(cfg.pytorch.gpu, async=True).float()
            target_var = target.cuda(cfg.pytorch.gpu, async=True).float()
            loss_msk_var = loss_msk.cuda(cfg.pytorch.gpu, async=True).float()
            trans_local_var = trans_local.cuda(cfg.pytorch.gpu, async=True).float()
            pose_var = pose.cuda(cfg.pytorch.gpu, async=True).float()
            c_box_var = c_box.cuda(cfg.pytorch.gpu, async=True).float()
            s_box_var = s_box.cuda(cfg.pytorch.gpu, async=True).float()
        else:
            inp_var = inp.float()
            target_var = target.float()
            loss_msk_var = loss_msk.float()
            trans_local_var = trans_local.float()
            pose_var = pose.float()
            c_box_var = c_box.float()
            s_box_var = s_box.float()

        bs = len(inp)
        # forward propagation
        T_begin = time.time()
        # import ipdb; ipdb.set_trace()
        (noc, w2d, scale), pred_trans = model(inp_var)
        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for a batch forward of resnet model is {}".format(T_end))

        if i % cfg.test.disp_interval == 0:
            # display input image
            inp_rgb = (inp[0].cpu().numpy().copy() * 255)[::-1, :, :].astype(np.uint8)
            cfg.writer.add_image('input_image', inp_rgb, i)
            cv2.imwrite(os.path.join(vis_dir, '{}_inp.png'.format(i)), inp_rgb.transpose(1,2,0)[:, :, ::-1])
            if 'rot' in cfg.pytorch.task.lower():
                # display coordinates map
                pred_coor = noc[0].data.cpu().numpy().copy()
                pred_coor[0] = im_norm_255(pred_coor[0])
                pred_coor[1] = im_norm_255(pred_coor[1])
                pred_coor[2] = im_norm_255(pred_coor[2])
                pred_coor = np.asarray(pred_coor, dtype=np.uint8)
                cfg.writer.add_image('train_coor_x_pred', np.expand_dims(pred_coor[0], axis=0), i)
                cfg.writer.add_image('train_coor_y_pred', np.expand_dims(pred_coor[1], axis=0), i)
                cfg.writer.add_image('train_coor_z_pred', np.expand_dims(pred_coor[2], axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_pred.png'.format(i)), pred_coor[0])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_pred.png'.format(i)), pred_coor[1])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_pred.png'.format(i)), pred_coor[2])
                gt_coor = target[0, 0:3].data.cpu().numpy().copy()
                gt_coor[0] = im_norm_255(gt_coor[0])
                gt_coor[1] = im_norm_255(gt_coor[1])
                gt_coor[2] = im_norm_255(gt_coor[2])
                gt_coor = np.asarray(gt_coor, dtype=np.uint8)
                cfg.writer.add_image('train_coor_x_gt', np.expand_dims(gt_coor[0], axis=0), i)
                cfg.writer.add_image('train_coor_y_gt', np.expand_dims(gt_coor[1], axis=0), i)
                cfg.writer.add_image('train_coor_z_gt', np.expand_dims(gt_coor[2], axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_x_gt.png'.format(i)), gt_coor[0])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_y_gt.png'.format(i)), gt_coor[1])
                cv2.imwrite(os.path.join(vis_dir, '{}_coor_z_gt.png'.format(i)), gt_coor[2])
                # display confidence map
                pred_conf = w2d[0].reshape(2, -1).softmax(dim=-1)
                pred_conf = pred_conf.mean(dim=0).reshape(64, 64).data.cpu().numpy().copy()
                pred_conf = (im_norm_255(pred_conf)).astype(np.uint8)
                cfg.writer.add_image('train_conf_pred', np.expand_dims(pred_conf, axis=0), i)
                cv2.imwrite(os.path.join(vis_dir, '{}_conf_pred.png'.format(i)), pred_conf)
            if 'trans' in cfg.pytorch.task.lower():
                pred_trans_ = pred_trans[0].data.cpu().numpy().copy()
                gt_trans_ = trans_local[0].data.cpu().numpy().copy()
                cfg.writer.add_scalar('train_trans_x_gt', gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_gt', gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_gt', gt_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_x_pred', pred_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_pred', pred_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_pred', pred_trans_[2], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_x_err', pred_trans_[0]-gt_trans_[0], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_y_err', pred_trans_[1]-gt_trans_[1], i + (epoch-1) * num_iters)
                cfg.writer.add_scalar('train_trans_z_err', pred_trans_[2]-gt_trans_[2], i + (epoch-1) * num_iters)

        # loss
        if 'rot' in cfg.pytorch.task.lower() and not cfg.network.rot_head_freeze:
            dim = [[abs(obj_info[obj_id_]['min_x']),
                    abs(obj_info[obj_id_]['min_y']),
                    abs(obj_info[obj_id_]['min_z'])] for obj_id_ in obj_id.cpu().numpy()]
            dim = noc.new_tensor(dim)  # (n, 3)
            x3d = noc * dim[..., None, None]  # (bs, 3, h, w)

            s = s_box_var.to(torch.int64)
            wh_begin = c_box_var.to(torch.int64) - s[:, None] / 2.  # (n, 2)
            wh_unit = s.to(torch.float32) / cfg.dataiter.out_res  # (n, )

            wh_arange = torch.arange(cfg.dataiter.out_res, device=noc.device, dtype=torch.float32)
            y, x = torch.meshgrid(wh_arange, wh_arange)  # (h, w)
            # (bs, 2, h, w)
            x2d = torch.stack((wh_begin[:, 0, None, None] + x * wh_unit[:, None, None],
                               wh_begin[:, 1, None, None] + y * wh_unit[:, None, None]), dim=1)
            rot_mat = pose_var[:, :, :3]  # (bs, 3, 3)
            trans_vec = pose_var[:, :, 3]  # (bs, 3)
            rot_quat = matrix_to_quaternion(rot_mat)
            pose_gt = torch.cat((trans_vec, rot_quat), dim=-1)

            sample_pts = [np.random.choice(64 * 64, size=64 * 64 // 8, replace=False) for _ in range(bs)]
            sample_inds = x2d.new_tensor(sample_pts, dtype=torch.int64)
            batch_inds = torch.arange(bs, device=x2d.device)[:, None]
            x3d = x3d.flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            x2d = x2d.flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            w2d = w2d.flatten(2).transpose(-1, -2)[batch_inds, sample_inds]
            # Due to a legacy design decision, we use an alternative to standard softmax, i.e., normalizing
            # the mean before exponential map.
            w2d = (w2d - w2d.mean(dim=1, keepdim=True) - math.log(w2d.size(1))).exp() * scale[:, None, :]
            # To use standard softmax, comment out the line above and uncomment the line below:
            # w2d = w2d.softmax(dim=1) * scale[:, None, :]

            allowed_border = 30 * wh_unit  # (n, )
            camera = PerspectiveCamera(
                cam_mats=cam_intrinsic[None].expand(bs, -1, -1),
                z_min=0.01,
                lb=wh_begin - allowed_border[:, None],
                ub=wh_begin + (cfg.dataiter.out_res - 1) * wh_unit[:, None] + allowed_border[:, None])
            cost_fun = AdaptiveHuberPnPCost(
                relative_delta=0.1)
            cost_fun.set_param(x2d, w2d)
            _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
                x3d, x2d, w2d, camera, cost_fun,
                pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)

            loss_mc = model.monte_carlo_pose_loss(
                pose_sample_logweights, cost_tgt, scale.detach().mean())

            loss_t = (pose_opt_plus[:, :3] - pose_gt[:, :3]).norm(dim=-1)
            beta = 0.05
            loss_t = torch.where(loss_t < beta, 0.5 * loss_t.square() / beta,
                                 loss_t - 0.5 * beta)
            loss_t = loss_t.mean()

            dot_quat = (pose_opt_plus[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
            loss_r = (1 - dot_quat.square()) * 2
            loss_r = loss_r.mean()

            loss_rot = criterions[cfg.loss.rot_loss_type](
                loss_msk_var[:, :3] * noc, loss_msk_var[:, :3] * target_var[:, :3])

        else:
            loss_mc = 0
            loss_t = 0
            loss_r = 0
            loss_rot = 0
        if 'trans' in cfg.pytorch.task.lower() and not cfg.network.trans_head_freeze:
            loss_trans = criterions[cfg.loss.trans_loss_type](pred_trans, trans_local_var)
        else:
            loss_trans = 0
        loss = cfg.loss.rot_loss_weight * loss_rot + cfg.loss.trans_loss_weight * loss_trans \
               + cfg.loss.mc_loss_weight * loss_mc + cfg.loss.t_loss_weight * loss_t \
               + cfg.loss.r_loss_weight * loss_r

        Loss.update(loss.item() if loss != 0 else 0, bs)
        Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs)
        Loss_trans.update(loss_trans.item() if loss_trans != 0 else 0, bs)
        Loss_mc.update(loss_mc.item() if loss_mc != 0 else 0, bs)
        Loss_t.update(loss_t.item() if loss_t != 0 else 0, bs)
        Loss_r.update(loss_r.item() if loss_r != 0 else 0, bs)
        Norm_factor.update(model.monte_carlo_pose_loss.norm_factor.item(), bs)

        cfg.writer.add_scalar('data/loss', loss.item() if loss != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_rot', loss_rot.item() if loss_rot != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_trans', loss_trans.item() if loss_trans != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_mc', loss_mc.item() if loss_mc != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_t', loss_t.item() if loss_t != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/loss_r', loss_r.item() if loss_r != 0 else 0, cur_iter)
        cfg.writer.add_scalar('data/norm_factor', model.monte_carlo_pose_loss.norm_factor.item(), cur_iter)

        optimizer.zero_grad()
        model.zero_grad()
        T_begin = time.time()
        loss.backward()

        grad_norm = []
        for p in model.parameters():
            if (p.grad is None) or (not p.requires_grad):
                continue
            else:
                grad_norm.append(torch.norm(p.grad.detach()))
        grad_norm = torch.norm(torch.stack(grad_norm))
        Grad_norm.update(grad_norm.item(), bs)
        cfg.writer.add_scalar('data/grad_norm', grad_norm.item(), cur_iter)

        if not torch.isnan(grad_norm).any():
            optimizer.step()

        T_end = time.time() - T_begin
        if time_monitor:
            logger.info("time for backward of model: {}".format(T_end))
       
        Bar.suffix = 'train Epoch: [{0}][{1}/{2}]| Total: {total:} | ETA: {eta:} | ' \
                     'Loss {loss.avg:.4f} | Loss_rot {loss_rot.avg:.4f} | ' \
                     'Loss_trans {loss_trans.avg:.4f} | Loss_mc {loss_mc.avg:.4f} | ' \
                     'Norm_factor {norm_factor.avg:.4f} | Grad_norm {grad_norm.avg:.4f} | ' \
                     'Loss_t {loss_t.avg:.4f} | Loss_r {loss_r.avg:.4f}'.format(
            epoch, i, num_iters, total=bar.elapsed_td, eta=bar.eta_td,
            loss=Loss, loss_rot=Loss_rot, loss_trans=Loss_trans, loss_mc=Loss_mc,
            norm_factor=Norm_factor, grad_norm=Grad_norm, loss_t=Loss_t, loss_r=Loss_r)
        bar.next()
    bar.finish()
    return {'Loss': Loss.avg, 'Loss_rot': Loss_rot.avg, 'Loss_trans': Loss_trans.avg,
            'Loss_mc': Loss_mc.avg, 'Norm_factor': Norm_factor.avg, 'Grad_norm': Grad_norm.avg,
            'Loss_t': Loss_t.avg, 'Loss_r': Loss_r.avg}, preds
