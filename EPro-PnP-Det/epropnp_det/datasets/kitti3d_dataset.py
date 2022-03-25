"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
import os.path as osp
import shutil

import numpy as np
from scipy.linalg import solve_triangular
import mmcv
from ..core import kitti_eval

from mmdet.datasets import CustomDataset, DATASETS


@DATASETS.register_module()
class KITTI3DDataset(CustomDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def __init__(self,
                 ann_file,
                 pipeline,
                 label_prefix=None,
                 meta_prefix=None,
                 coord_3d_prefix=None,
                 calib_prefix=None,
                 depth_prefix=None,
                 calib_cam=2,
                 max_truncation=0.95,
                 max_occlusion=3,
                 min_height=8,
                 **kwargs):
        self.label_prefix = label_prefix
        self.labels = []
        self.meta_prefix = meta_prefix
        self.coord_3d_prefix = coord_3d_prefix
        self.calib_prefix = calib_prefix
        self.depth_prefix = depth_prefix
        self.calibs = []
        self.calib_cam = calib_cam
        self.max_truncation = max_truncation
        self.max_occlusion = max_occlusion
        self.min_height = min_height
        super(KITTI3DDataset, self).__init__(
            ann_file,
            pipeline,
            **kwargs)

    @staticmethod
    def open_calib_file(calib_file, cam):
        assert 0 <= cam <= 3
        f_calib = open(calib_file).readlines()[cam]
        proj_mat = np.array(
            [float(v) for v in f_calib.strip().split(' ')[1:]], dtype=np.float32
        ).reshape((3, 4))
        return proj_mat

    @staticmethod
    def open_label_file(path):
        f_label = open(path)
        label = [[float(v) if i != 0 and i != 2
                  else int(v) if i == 2 else v
                  for i, v in enumerate(line_label.strip().split(' '))]
                 for line_label in f_label]
        return label

    def load_annotations(self, ann_file):
        filenames = open(ann_file).readlines()
        data_infos = []
        for filename in filenames:
            filename_raw = filename.strip()
            if self.meta_prefix is not None:
                meta_path = osp.join(self.meta_prefix, filename_raw + '.txt')
                height, width = np.loadtxt(meta_path, delimiter=',')
            else:
                img_path = osp.join(self.img_prefix, filename_raw + '.png')
                img = mmcv.imread(img_path)
                height, width = img.shape[:2]
            cali_mat = self.open_calib_file(
                osp.join(self.calib_prefix, filename_raw + '.txt'), self.calib_cam)
            cali_t_vec = cali_mat[:, 3:]
            cam_intrinsic = cali_mat[:, :3]
            cam_t_vec = solve_triangular(
                cam_intrinsic, cali_t_vec, lower=False).squeeze(-1)
            info = dict(filename=filename_raw + '.png',
                        width=int(width),
                        height=int(height),
                        cam_intrinsic=cam_intrinsic,
                        cam_t_vec=cam_t_vec)
            data_infos.append(info)
            if self.label_prefix is not None:
                label = self.open_label_file(
                    osp.join(self.label_prefix, filename_raw + '.txt'))
                self.labels.append(label)
        return data_infos

    def get_ann_info(self, idx):
        return self._parse_ann_info(idx)

    def pre_pipeline(self, results):
        super(KITTI3DDataset, self).pre_pipeline(results)
        results['coord_3d_prefix'] = self.coord_3d_prefix
        results['depth_prefix'] = self.depth_prefix
        results['bbox_3d_fields'] = []
        results['dense_fields'] = []

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if self.filter_empty_gt and len(self.get_ann_info(i)['bboxes']) == 0:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_cat_ids(self, idx):
        return self._parse_ann_info(idx)['labels'].astype(np.int).tolist()

    def _parse_ann_info(self, idx):
        object_ids = []
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_truncation = []
        gt_occlusion = []
        gt_alpha = []
        gt_bboxes_3d = []

        label = self.labels[idx]
        img_info = self.data_infos[idx]

        for object_id, instance in enumerate(label):
            instance_class = instance[0]
            truncation = instance[1]
            occlusion = instance[2]
            alpha = instance[3]
            bbox = instance[4:8]  # x1, y1, x2, y2
            bbox_3d = instance[8:]  # h, w, l, x, y, z, yaw

            if instance_class in self.CLASSES:
                height = bbox[3] - bbox[1]
                if truncation > self.max_truncation \
                        or occlusion > self.max_occlusion or height < self.min_height:
                    gt_bboxes_ignore.append(bbox)
                    continue
                object_ids.append(object_id)
                gt_labels.append(self.CLASSES.index(instance_class))  # zero-based
                gt_truncation.append(truncation)
                gt_occlusion.append(occlusion)
                gt_alpha.append(alpha)
                gt_bboxes.append(bbox)
                gt_bboxes_3d.append(bbox_3d)

            elif instance_class.lower() == 'dontcare':
                gt_bboxes_ignore.append(bbox)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_bboxes_3d[:, [0, 1, 2]] = gt_bboxes_3d[:, [2, 0, 1]]  # to lhw
            object_ids = np.array(object_ids, dtype=np.int)
        else:
            gt_bboxes = np.empty((0, 4), dtype=np.float32)
            gt_labels = np.empty(0, dtype=np.int64)
            gt_bboxes_3d = np.empty((0, 7), dtype=np.float32)
            object_ids = np.empty(0, dtype=np.int)

        gt_bboxes = np.minimum(
            gt_bboxes.clip(min=0),
            np.array([img_info['width'], img_info['height'], img_info['width'], img_info['height']],
                     dtype=np.float32))
        gt_bboxes_3d_eval = gt_bboxes_3d.copy()  # bboxes in reference space (for eval)
        gt_bboxes_3d[:, 3:6] += self.data_infos[idx]['cam_t_vec']  # move to camera space
        gt_bboxes_3d[:, 4] -= gt_bboxes_3d[:, 1] / 2  # center offset

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.empty((0, 4), dtype=np.float32)

        coord_3d = self.data_infos[idx]['filename'].replace('png', 'pkl')
        depth = self.data_infos[idx]['filename']

        ann = dict(
            object_ids=object_ids,
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            coord_3d=coord_3d,
            depth=depth,
            truncation=gt_truncation,
            occlusion=gt_occlusion,
            alpha=gt_alpha,
            bboxes_3d=gt_bboxes_3d,
            bboxes_3d_eval=gt_bboxes_3d_eval)
        return ann

    def evaluate(self,
                 results,
                 metric=['bbox', 'bev', '3d'],
                 logger=None,
                 out_dir=None,
                 summary_file=None,
                 print_summary=True,
                 use_r40=True):
        results = self.format_results(results, out_dir)
        gt_annos = [self.format_gt_anno(self.get_ann_info(i)) for i in range(len(self))]
        criteria = 'R40' if use_r40 else 'R11'
        ap_result_str, ap_dict = kitti_eval(
            gt_annos,
            results,
            self.CLASSES,
            eval_types=metric,
            criteria=criteria)
        if print_summary:
            print('\n' + ap_result_str)
        if summary_file is not None:
            f = open(summary_file, 'w')
            f.write(ap_result_str)
            f.close()
        return ap_dict

    def format_results(self, results, out_dir=None):
        det_annos = []
        for idx, result in enumerate(results):
            bbox_3d_results = result['bbox_3d_results']
            bbox_results = result['bbox_results']

            name = np.array([self.CLASSES[i]
                             for i, det_per_class in enumerate(bbox_results)
                             for _ in det_per_class])
            num_dets = name.shape[0]

            bbox_results_all = np.concatenate(bbox_results, axis=0)
            bbox_3d_results_all = np.concatenate(bbox_3d_results, axis=0)
            bbox_3d_results_all[:, 3:6] -= self.data_infos[idx]['cam_t_vec']
            bbox_3d_results_all[:, 4] += bbox_3d_results_all[:, 1] / 2

            sort_idx = bbox_3d_results_all[:, 7].argsort()[::-1]
            bbox_results_all_sorted = bbox_results_all[sort_idx]
            bbox_3d_results_all_sorted = bbox_3d_results_all[sort_idx]

            name = name[sort_idx]
            bbox = bbox_results_all_sorted[:, :4]
            dimensions = bbox_3d_results_all_sorted[:, :3]  # lhw
            location = bbox_3d_results_all_sorted[:, 3:6]
            rotation_y = bbox_3d_results_all_sorted[:, 6]
            score = bbox_3d_results_all_sorted[:, 7]
            # alpha is defined in LiDAR frame, hence +0.27
            alpha = rotation_y - np.arctan2(location[:, 0], location[:, 2] + 0.27)

            anno = dict(
                name=name,
                truncated=np.full(num_dets, -1, dtype=np.int8),
                occluded=np.full(num_dets, -1, dtype=np.int8),
                alpha=alpha,
                bbox=bbox,
                dimensions=dimensions,
                location=location,
                rotation_y=rotation_y,
                score=score)
            det_annos.append(anno)

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
            self.write_result_files(det_annos, osp.join(out_dir, 'data'))
        return det_annos

    def format_gt_anno(self, ann_info):
        num_objects = len(ann_info['bboxes'])
        num_dontcares = len(ann_info['bboxes_ignore'])
        num_gt = num_objects + num_dontcares
        anno = dict(
            name=[self.CLASSES[label] for label in ann_info['labels']
                  ] + ['DontCare'] * num_dontcares,
            truncated=np.array(ann_info['truncation']
                               + [-1] * num_dontcares, dtype=np.float32),
            occluded=np.array(ann_info['occlusion']
                              + [-1] * num_dontcares, dtype=np.float32),
            alpha=np.array(ann_info['alpha']
                           + [-10] * num_dontcares, dtype=np.float32),
            bbox=np.concatenate(
                (ann_info['bboxes'], ann_info['bboxes_ignore']), axis=0),
            dimensions=np.concatenate(
                (ann_info['bboxes_3d_eval'][:, :3],
                 np.full((num_dontcares, 3), -1, dtype=np.float32)),
                axis=0),
            location=np.concatenate(
                (ann_info['bboxes_3d_eval'][:, 3:6],
                 np.full((num_dontcares, 3), -1000, dtype=np.float32)),
                axis=0),
            rotation_y=np.concatenate(
                (ann_info['bboxes_3d_eval'][:, 6],
                 np.full(num_dontcares, -10, dtype=np.float32)),
                axis=0),
            score=np.zeros(num_gt, dtype=np.float32),
            index=np.concatenate(
                (np.arange(num_objects, dtype=np.int32),
                 np.full(num_dontcares, -1, dtype=np.int32)),
                axis=0),
            group_ids=np.arange(num_gt, dtype=np.int32))
        return anno

    def write_result_files(self, results, result_dir):
        if osp.exists(result_dir):
            shutil.rmtree(result_dir)
        os.mkdir(result_dir)
        for result, data_info in zip(results, self.data_infos):
            filename, _ = osp.splitext(data_info['filename'])
            all_result = np.concatenate(
                (result['name'].reshape(-1, 1),
                 result['truncated'].reshape(-1, 1),
                 result['occluded'].reshape(-1, 1),
                 result['alpha'].reshape(-1, 1),
                 result['bbox'],
                 result['dimensions'][:, [1, 2, 0]],  # to hwl
                 result['location'],
                 result['rotation_y'].reshape(-1, 1),
                 result['score'].reshape(-1, 1)),
                axis=1)
            filepath = osp.join(result_dir, filename + '.txt')
            np.savetxt(filepath, all_result, delimiter=' ', fmt='%s')
