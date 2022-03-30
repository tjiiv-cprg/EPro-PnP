"""
@author: Zhigang Li
@license: (C) Copyright.
@contact: aaalizhigang@163.com
@software: Pose6D
@file: LineMOD.py
@time: 18-10-24 下午10:24
@desc: load LineMOD dataset
"""

import torch.utils.data as data
import numpy as np
import ref
import cv2
from utils.img import zoom_in, get_edges, xyxy_to_xywh, Crop_by_Pad
from utils.transform3d import prj_vtx_cam
from utils.io import read_pickle
import os, sys
from tqdm import tqdm
import utils.fancy_logger as logger
import pickle
from glob import glob 
import random 
from utils.eval import calc_rt_dist_m 

class LM(data.Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        self.infos = self.load_lm_model_info(ref.lm_model_info_pth)
        self.cam_K = ref.K
        logger.info('==> initializing {} {} data.'.format(cfg.dataset.name, split))
        # load dataset
        annot = []
        if split == 'test':
            cache_dir = os.path.join(ref.cache_dir, 'test')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            for obj in tqdm(self.cfg.dataset.classes):
                cache_pth = os.path.join(cache_dir, '{}.npy'.format(obj))
                if not os.path.exists(cache_pth):
                    annot_cache = []
                    rgb_pths = glob(os.path.join(ref.lm_test_dir, obj, '*-color.png'))
                    for rgb_pth in tqdm(rgb_pths):
                        item = self.col_test_item(rgb_pth)
                        item['obj'] = obj
                        annot_cache.append(item)
                    np.save(cache_pth, annot_cache)
                annot.extend(np.load(cache_pth, allow_pickle=True).tolist())
            self.num = len(annot)
            self.annot = annot
            logger.info('load {} test samples.'.format(self.num))
        elif split == 'train':
            if 'real' in self.cfg.dataset.img_type:
                cache_dir = os.path.join(ref.cache_dir, 'train/real')
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                for obj in tqdm(self.cfg.dataset.classes):
                    cache_pth = os.path.join(cache_dir, '{}.npy'.format(obj))
                    if not os.path.exists(cache_pth):
                        annot_cache = []
                        rgb_pths = glob(os.path.join(ref.lm_train_real_dir, obj, '*-color.png'))
                        for rgb_pth in tqdm(rgb_pths):
                            item = self.col_train_item(rgb_pth)
                            item['obj'] = obj
                            annot_cache.append(item)
                        np.save(cache_pth, annot_cache)
                    annot.extend(np.load(cache_pth, allow_pickle=True).tolist())
                self.real_num = len(annot)
                logger.info('load {} real training samples.'.format(self.real_num))
            if 'imgn' in self.cfg.dataset.img_type:
                cache_dir = os.path.join(ref.cache_dir, 'train/imgn')
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
                for obj in tqdm(self.cfg.dataset.classes):
                    cache_pth = os.path.join(cache_dir, '{}.npy'.format(obj))
                    if not os.path.exists(cache_pth):
                        annot_cache = []
                        coor_pths = sorted(glob(os.path.join(ref.lm_train_imgn_dir, obj, '*-coor.pkl')))
                        for coor_pth in tqdm(coor_pths):
                            item = self.col_imgn_item(coor_pth)
                            item['obj'] = obj
                            annot_cache.append(item)
                        np.save(cache_pth, annot_cache)
                    annot_obj = np.load(cache_pth, allow_pickle=True).tolist()
                    annot_obj_num = len(annot_obj)
                    if (annot_obj_num > self.cfg.dataset.syn_num) and (self.cfg.dataset.syn_samp_type != ''):
                        if self.cfg.dataset.syn_samp_type == 'uniform':
                            samp_idx = np.linspace(0, annot_obj_num - 1, self.cfg.dataset.syn_num, dtype=np.int32)
                        elif self.cfg.dataset.syn_samp_type == 'random':
                            samp_idx = random.sample(range(annot_obj_num), self.cfg.dataset.syn_num)
                        else:
                            raise ValueError
                        annot_obj = np.asarray(annot_obj)[samp_idx].tolist()
                    annot.extend(annot_obj)
                self.imgn_num = len(annot) - self.real_num
                logger.info('load {} imgn training samples.'.format(self.imgn_num))
            self.num = len(annot)
            self.annot = annot
            logger.info('load {} training samples, including {} real samples and {} synthetic samples.'.format(self.num, self.real_num, self.imgn_num))
            self.bg_list = self.load_bg_list()
        else:
            raise ValueError
        
    def col_test_item(self, rgb_pth):
        item = {}
        item['rgb_pth'] = rgb_pth
        item['pose'] = np.loadtxt(rgb_pth.replace('-color.png', '-pose.txt'))
        item['box'] = xyxy_to_xywh(np.loadtxt(rgb_pth.replace('-color.png', '-box_fasterrcnn.txt')))
        item['mask_pth'] = rgb_pth.replace('-color.png', '-label.png')
        item['coor_pth'] = rgb_pth.replace('-color.png', '-coor.pkl')
        item['data_type'] = 'real'
        return item 

    def col_train_item(self, rgb_pth):
        item = {}
        item['rgb_pth'] = rgb_pth
        item['pose'] = np.loadtxt(rgb_pth.replace('-color.png', '-pose.txt'))
        item['box'] = np.loadtxt(rgb_pth.replace('-color.png', '-box.txt'))
        item['mask_pth'] = rgb_pth.replace('-color.png', '-label.png')
        item['coor_pth'] = rgb_pth.replace('-color.png', '-coor.pkl')
        item['data_type'] = 'real'
        return item 

    def col_imgn_item(self, coor_pth):
        item = {}
        item['coor_pth'] = coor_pth
        item['rgb_pth'] = coor_pth.replace('-coor.pkl', '-color.png')
        try:
            item['pose'] = np.loadtxt(coor_pth.replace('-coor.pkl', '-pose.txt'))
        except:
            print('coor_pth: {}'.format(coor_pth))
            raise
        item['box'] = np.loadtxt(coor_pth.replace('-coor.pkl', '-box.txt'))
        item['mask_pth'] = coor_pth.replace('-coor.pkl', '-label.png')
        item['coor_pth'] = coor_pth.replace('-coor.pkl', '-coor.pkl')
        item['data_type'] = 'imgn'
        return item 

    @staticmethod
    def load_lm_model_info(info_pth):
        infos = {}
        with open(info_pth, 'r') as f:
            for line in f.readlines():
                items = line.strip().split(' ')
                cls_idx = int(items[0])
                infos[cls_idx] = {}
                infos[cls_idx]['diameter'] = float(items[2]) / 1000. # unit: mm => m
                infos[cls_idx]['min_x'] = float(items[4]) / 1000.
                infos[cls_idx]['min_y'] = float(items[6]) / 1000.
                infos[cls_idx]['min_z'] = float(items[8]) / 1000.
        return infos

    @staticmethod
    def load_bg_list():
        path = os.path.join(ref.bg_dir, 'VOC2012/ImageSets/Main/diningtable_trainval.txt')
        with open(path, 'r') as f:
            bg_list = [line.strip('\r\n').split()[0] for line in f.readlines() if
                                line.strip('\r\n').split()[1] == '1']
        return bg_list

    @staticmethod
    def load_bg_im(im_real, bg_list):
        h, w, c = im_real.shape
        bg_num = len(bg_list)
        idx = random.randint(0, bg_num - 1)
        bg_path = os.path.join(ref.bg_dir, 'VOC2012/JPEGImages/{}.jpg'.format(bg_list[idx]))
        bg_im = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        bg_h, bg_w, bg_c = bg_im.shape
        real_hw_ratio = float(h) / float(w)
        bg_hw_ratio = float(bg_h) / float(bg_w)
        if real_hw_ratio <= bg_hw_ratio:
            crop_w = bg_w
            crop_h = int(bg_w * real_hw_ratio)
        else:
            crop_h = bg_h 
            crop_w = int(bg_h / bg_hw_ratio)
        bg_im = bg_im[:crop_h, :crop_w, :]
        bg_im = cv2.resize(bg_im, (w, h), interpolation=cv2.INTER_LINEAR)
        return bg_im
        
    def change_bg(self, rgb, msk):
        """
        change image's background
        """
        bg_im = self.load_bg_im(rgb, self.bg_list)
        msk = np.dstack([msk, msk, msk]).astype(np.bool)
        bg_im[msk] = rgb[msk]
        return bg_im

    def load_obj(self, idx):
        return self.annot[idx]['obj']

    def load_type(self, idx):
        return self.annot[idx]['data_type']

    def load_pose(self, idx):
        return self.annot[idx]['pose']

    def load_box(self, idx):
        return self.annot[idx]['box']

    def load_msk_syn(self, idx):
        return cv2.imread(self.annot[idx]['mask_pth'], cv2.IMREAD_GRAYSCALE)

    def load_msk_real(self, idx, obj_id):
        return (cv2.imread(self.annot[idx]['mask_pth'], cv2.IMREAD_GRAYSCALE) == obj_id).astype(np.uint8)

    def load_rgb(self, idx):
        return cv2.imread(self.annot[idx]['rgb_pth'])

    def load_coor(self, idx, restore=True, coor_h=480, coor_w=640):
        try:
            coor_load = read_pickle(self.annot[idx]['coor_pth'])
        except:
            print('coor_pth: {}'.format(self.annot[idx]['coor_pth']))
            raise
        if not restore:
            return coor_load['coor']
        else:
            u = coor_load['u']
            l = coor_load['l']
            h = coor_load['h']
            w = coor_load['w']
            coor = np.zeros((coor_h, coor_w, 3)).astype(np.float32)
            coor[u:(u+h),l:(l+w),:] = coor_load['coor']
            return coor

    def xywh_to_cs_dzi(self, xywh, s_ratio, s_max=None, tp='uniform'):
        x, y, w, h = xywh
        if tp == 'gaussian':
            sigma = 1
            shift = truncnorm.rvs(-self.cfg.augment.shift_ratio / sigma, self.cfg.augment.shift_ratio / sigma, scale=sigma, size=2)
            scale = 1+truncnorm.rvs(-self.cfg.augment.scale_ratio / sigma, self.cfg.augment.scale_ratio / sigma, scale=sigma, size=1)
        elif tp == 'uniform':
            scale = 1+self.cfg.augment.scale_ratio * (2*np.random.random_sample()-1)
            shift = self.cfg.augment.shift_ratio * (2*np.random.random_sample(2)-1)
        else:
            raise
        c = np.array([x+w*(0.5+shift[1]), y+h*(0.5+shift[0])]) # [c_w, c_h]
        s = max(w, h)*s_ratio*scale
        if s_max != None:
            s = min(s, s_max)
        return c, s

    @staticmethod
    def xywh_to_cs(xywh, s_ratio, s_max=None):
        x, y, w, h = xywh
        c = np.array((x+0.5*w, y+0.5*h)) # [c_w, c_h]
        s = max(w, h)*s_ratio
        if s_max != None:
            s = min(s, s_max)
        return c, s

    def denoise_coor(self, coor):
        """
        denoise coordinates by median blur
        """
        coor_blur = cv2.medianBlur(coor, 3)
        edges = get_edges(coor)
        coor[edges != 0] = coor_blur[edges != 0]
        return coor

    def norm_coor(self, coor, obj_id):
        """
        normalize coordinates by object size
        """
        coor_x, coor_y, coor_z = coor[..., 0], coor[..., 1], coor[..., 2]
        coor_x = coor_x / abs(self.infos[obj_id]['min_x'])
        coor_y = coor_y / abs(self.infos[obj_id]['min_y'])
        coor_z = coor_z / abs(self.infos[obj_id]['min_z'])
        return np.dstack((coor_x, coor_y, coor_z))

    def c_rel_delta(self, c_obj, c_box, wh_box):
        """
        compute relative bias between object center and bounding box center
        """
        c_delta = np.asarray(c_obj) - np.asarray(c_box)
        c_delta /= np.asarray(wh_box)
        return c_delta

    def d_scaled(self, depth, s_box, res):
        """
        compute scaled depth
        """
        r = float(res) / s_box
        return depth / r

    def __getitem__(self, idx):
        if self.split == 'train':
            obj = self.load_obj(idx)
            obj_id = ref.obj2idx(obj)
            data_type = self.load_type(idx)
            box = self.load_box(idx)
            pose = self.load_pose(idx)
            rgb = self.load_rgb(idx)
            if data_type == 'real':
                msk = self.load_msk_real(idx, obj_id)
            else:
                msk = self.load_msk_syn(idx)
            coor = self.load_coor(idx)
            if self.split == 'train':
                if (self.annot[idx]['data_type']=='imgn') or (random.random()<self.cfg.augment.change_bg_ratio):
                    rgb = self.change_bg(rgb, msk)
            if (self.split == 'train') and self.cfg.dataiter.dzi:
                c, s = self.xywh_to_cs_dzi(box, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            else:
                c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            if self.cfg.dataiter.denoise_coor:
                coor = self.denoise_coor(coor)

            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            msk, *_ = zoom_in(msk, c, s, self.cfg.dataiter.out_res, channel=1)
            coor, *_ = zoom_in(coor, c, s, self.cfg.dataiter.out_res, interpolate=cv2.INTER_NEAREST)
            c = np.array([c_w_, c_h_])
            s = s_
            coor = self.norm_coor(coor, obj_id).transpose(2, 0, 1)
            inp = rgb
            out = np.concatenate([coor, msk[None, :, :]], axis=0)
            loss_msk = np.stack([msk, msk, msk, np.ones_like(msk)], axis=0)
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, self.cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)
            return obj, obj_id, inp, out, loss_msk, trans_local, pose, c, s, np.asarray(box)

        if self.split == 'test':
            obj = self.load_obj(idx)
            obj_id = ref.obj2idx(obj)
            box = self.load_box(idx)
            pose = self.load_pose(idx)
            rgb = self.load_rgb(idx)
            c, s = self.xywh_to_cs(box, self.cfg.augment.pad_ratio, s_max=max(ref.im_w, ref.im_h))
            rgb, c_h_, c_w_, s_ = zoom_in(rgb, c, s, self.cfg.dataiter.inp_res)
            rgb = rgb.transpose(2, 0, 1).astype(np.float32) / 255.
            c = np.array([c_w_, c_h_])
            s = s_
            inp = rgb
            trans = pose[:, 3]
            c_obj, _ = prj_vtx_cam(trans, self.cam_K)
            c_delta = self.c_rel_delta(c_obj, c, box[2:])
            d_local = self.d_scaled(trans[2], s, self.cfg.dataiter.out_res)
            trans_local = np.append(c_delta, [d_local], axis=0)
            return obj, obj_id, inp, pose, c, s, np.asarray(box), trans_local

        if False:
            # check trans_local 
            ratio_delta_c = trans_local[:2]
            ratio_depth = trans_local[2]
            pred_depth = ratio_depth * (self.cfg.dataiter.out_res / s)
            pred_c = ratio_delta_c * box[2:] + c
            pred_x = (pred_c[0] - self.cam_K[0, 2]) * pred_depth / self.cam_K[0, 0]
            pred_y = (pred_c[1] - self.cam_K[1, 2]) * pred_depth / self.cam_K[1, 1]
            T_vector_trans = np.asarray([pred_x, pred_y, pred_depth])
            pose_est_trans = np.concatenate([np.eye(3), np.asarray((T_vector_trans).reshape(3, 1))], axis=1)

        if False:
            # check coordinates
            pred_coor_ = coor.transpose(1,2,0)
            pred_coor_[:, :, 0] = pred_coor_[:, :, 0] * abs(self.infos[obj_id]['min_x'])
            pred_coor_[:, :, 1] = pred_coor_[:, :, 1] * abs(self.infos[obj_id]['min_y'])
            pred_coor_[:, :, 2] = pred_coor_[:, :, 2] * abs(self.infos[obj_id]['min_z'])
            pred_coor_ = pred_coor_.tolist()
            pred_conf_ = ((msk - msk.min()) / (msk.max() - msk.min())).tolist()
            select_pts_2d = []
            select_pts_3d = []
            c_w = int(c[0])
            c_h = int(c[1])
            w_begin = c_w - int(s) / 2.
            h_begin = c_h - int(s) / 2.
            w_unit = int(s) * 1.0 / self.cfg.dataiter.out_res
            h_unit = int(s) * 1.0 / self.cfg.dataiter.out_res
            min_x = 0.001 * abs(self.infos[obj_id]['min_x'])
            min_y = 0.001 * abs(self.infos[obj_id]['min_y'])
            min_z = 0.001 * abs(self.infos[obj_id]['min_z'])
            for x in range(self.cfg.dataiter.out_res):
                for y in range(self.cfg.dataiter.out_res):
                    if pred_conf_[x][y] < self.cfg.test.mask_threshold:
                        continue
                    if abs(pred_coor_[x][y][0]) < min_x  and abs(pred_coor_[x][y][1]) < min_y  and \
                        abs(pred_coor_[x][y][2]) < min_z:
                        continue
                    select_pts_2d.append([w_begin + y * w_unit, h_begin + x * h_unit])
                    select_pts_3d.append(pred_coor_[x][y])

            model_points = np.asarray(select_pts_3d, dtype=np.float32)
            image_points = np.asarray(select_pts_2d, dtype=np.float32)
            dist_coeffs = np.zeros((4, 1)) 
            try:
                _, R_vector, T_vector, inliers = cv2.solvePnPRansac(model_points, image_points, self.cam_K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
                R_matrix = cv2.Rodrigues(R_vector, jacobian=0)[0]
                pose_est = np.concatenate((R_matrix, np.asarray(T_vector).reshape(3, 1)), axis=1)
                r_err, t_err = calc_rt_dist_m(pose_est, pose)
                print('len corres: {}'.format(len(model_points)))
                print('pose_gt: {}\npose_est: {}\n rot_err: {} trans_err: {}\n'.format(pose, pose_est, r_err, t_err))
            except:
                pass

    def __len__(self):
        return self.num
