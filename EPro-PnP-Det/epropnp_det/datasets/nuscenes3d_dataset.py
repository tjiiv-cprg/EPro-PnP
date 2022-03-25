"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/open-mmlab/mmdetection3d
"""

import os.path as osp
import tempfile
import warnings

import numpy as np
import torch
import mmcv
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.config import config_factory
from pyquaternion import Quaternion

from mmdet.datasets import CustomDataset, DATASETS

from epropnp_det.ops.iou3d.iou3d_utils import nms_gpu
from ..core import rot_mat_to_yaw


@DATASETS.register_module()
class NuScenes3DDataset(CustomDataset):
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')
    CAMS = ('CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT')
    NUM_CAMS = len(CAMS)
    KITTI2NUS_ROT = np.array([[1,  0, 0],
                              [0,  0, 1],
                              [0, -1, 0]], dtype=np.float32)
    Attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', '')
    CLS_ORIENTATION = [True, True, True, True, True, True, True, True, False, False]
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLS2ATTR = {
        'car': ('vehicle.moving', 'vehicle.parked', 'vehicle.stopped'),
        'truck': ('vehicle.moving', 'vehicle.parked', 'vehicle.stopped'),
        'trailer': ('vehicle.moving', 'vehicle.parked', 'vehicle.stopped'),
        'bus': ('vehicle.moving', 'vehicle.parked', 'vehicle.stopped'),
        'construction_vehicle': ('vehicle.moving', 'vehicle.parked', 'vehicle.stopped'),
        'bicycle': ('cycle.with_rider', 'cycle.without_rider'),
        'motorcycle': ('cycle.with_rider', 'cycle.without_rider'),
        'pedestrian': ('pedestrian.moving', 'pedestrian.standing', 'pedestrian.sitting_lying_down'),
        'traffic_cone': ('', ),
        'barrier': ('', )
    }

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix=None,
                 trunc_ignore_thres=0.8,
                 min_box_size=4.0,
                 min_visibility=2,
                 path_mapping=None,
                 nms_thr=0.25,
                 eval_version='detection_cvpr_2019',
                 step=1,  # only for debug & visualization purpose
                 max_num_samples=-1,  # only for debug purpose
                 **kwargs):
        self.trunc_ignore_thres = trunc_ignore_thres
        self.min_box_size = min_box_size
        self.min_visibility = min_visibility
        self.path_mapping = path_mapping
        self.step = step
        self.max_num_samples = max_num_samples
        self.nms_thr = nms_thr
        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)
        self.eval_detection_configs.class_names = list(self.eval_detection_configs.class_names)
        self.modality = dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
            use_map=False,
            use_external=False)
        super(NuScenes3DDataset, self).__init__(
            ann_file,
            pipeline,
            data_root=data_root,
            img_prefix=img_prefix,
            **kwargs)

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file)
        data_infos = []
        for frame_info in data['infos']:
            sample_token = frame_info['token']
            for cam_id, cam in enumerate(self.CAMS):
                cam_info = frame_info['cams'][cam]
                data_path = cam_info['data_path']
                oc_path = cam_info['oc_path']
                if self.path_mapping is not None:
                    for old, new in self.path_mapping.items():
                        data_path = data_path.replace(old, new)
                        oc_path = oc_path.replace(old, new)
                data_infos.append(dict(
                    filename=data_path,
                    width=cam_info['imsize'][0],
                    height=cam_info['imsize'][1],
                    cam_id=cam_id,
                    cam_intrinsic=cam_info['cam_intrinsic'].astype(np.float32),
                    ann_records=cam_info['ann_records'],
                    oc_path=oc_path,
                    sample_token=sample_token,
                    sensor2ego_translation=cam_info['sensor2ego_translation'],
                    sensor2ego_rotation=cam_info['sensor2ego_rotation'],
                    ego2global_translation=cam_info['ego2global_translation'],
                    ego2global_rotation=cam_info['ego2global_rotation']
                ))
        end = len(data_infos)
        if self.max_num_samples >= 0:
            end = min(end, self.max_num_samples * self.step)
        data_infos = data_infos[:end:self.step]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_ann_info(self, idx):
        return self._parse_ann_info(self.data_infos[idx])

    def pre_pipeline(self, results):
        super().pre_pipeline(results)
        results['bbox_3d_fields'] = []
        results['dense_fields'] = []
        results['coord_3d_prefix'] = ''

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
        return self.get_ann_info(idx)['labels'].astype(np.int).tolist()

    def _parse_ann_info(self, data_info):
        gt_bboxes = []
        gt_bboxes_ignore = []
        gt_labels = []
        gt_attr = []
        gt_velo = []
        gt_truncation = []
        gt_visibility = []
        gt_bboxes_3d = []

        object_ids = []

        for object_id, ann_record in enumerate(data_info['ann_records']):
            visibility = int(ann_record['visibility'])
            truncation = ann_record['truncation']
            # visibility and class filtering
            if visibility >= self.min_visibility and ann_record['cat_name'] in self.CLASSES:
                bbox = np.array(ann_record['bbox'], dtype=np.float32)
                wh = bbox[2:] - bbox[:2]
                # truncation & size filtering
                if truncation <= self.trunc_ignore_thres and wh.min() >= self.min_box_size:
                    gt_bboxes.append(bbox)
                    gt_labels.append(ann_record['cat_id'])
                    gt_attr.append(ann_record['attr_id'])
                    gt_velo.append(np.array(ann_record['velo'], dtype=np.float32))
                    gt_truncation.append(truncation)
                    gt_visibility.append(visibility)
                    object_ids.append(object_id)
                    # convert 3d box into KITTI format
                    bbox3d = ann_record['bbox3d']
                    lhw = bbox3d.wlh[[1, 2, 0]].astype(np.float32)
                    center = bbox3d.center.astype(np.float32)
                    rotation_matrix = bbox3d.rotation_matrix @ self.KITTI2NUS_ROT
                    yaw = rot_mat_to_yaw(rotation_matrix).astype(np.float32)
                    gt_bboxes_3d.append(np.concatenate(
                        [lhw, center, [yaw]]))
                else:
                    gt_bboxes_ignore.append(bbox)

        if gt_bboxes:
            gt_bboxes = np.stack(gt_bboxes, axis=0)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_attr = np.array(gt_attr, dtype=np.int64)
            gt_velo = np.stack(gt_velo, axis=0)
            gt_bboxes_3d = np.stack(gt_bboxes_3d, axis=0)
            object_ids = np.array(object_ids, dtype=np.int)
        else:
            gt_bboxes = np.empty((0, 4), dtype=np.float32)
            gt_labels = np.empty(0, dtype=np.int64)
            gt_attr = np.empty(0, dtype=np.int64)
            gt_velo = np.empty((0, 2), dtype=np.float32)
            gt_bboxes_3d = np.empty((0, 7), dtype=np.float32)
            object_ids = np.empty(0, dtype=np.int)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.stack(gt_bboxes_ignore, axis=0)
        else:
            gt_bboxes_ignore = np.empty((0, 4), dtype=np.float32)

        ann = dict(
            object_ids=object_ids,
            bboxes=gt_bboxes,
            labels=gt_labels,
            attr=gt_attr,
            velo=gt_velo,
            bboxes_ignore=gt_bboxes_ignore,
            coord_3d=data_info['oc_path'],
            coord_3d_rot=self.KITTI2NUS_ROT.T.astype(np.float32),
            truncation=gt_truncation,
            visibility=gt_visibility,
            bboxes_3d=gt_bboxes_3d)
        return ann

    def evaluate(self,
                 results,
                 metric='NDS',
                 logger=None,
                 jsonfile_prefix=None):
        if not ('NDS' in metric or 'nds' in metric):
            warnings.warn('Evaluation metric should be specified')
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        results_dict = self._evaluate_single(result_files)
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return results_dict

    def _evaluate_single(self,
                         result_path,
                         result_name='pts_bbox'):
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        assert len(results) % self.NUM_CAMS == 0
        frames = [[] for _ in range(len(results) // self.NUM_CAMS)]
        frame_results = []
        for i, (result, data_info) in enumerate(zip(results, self.data_infos)):
            frame_id = i // self.NUM_CAMS
            result.update(data_info)
            frames[frame_id].append(result)
        print('\nConverting to surround results...')
        for frame in mmcv.track_iter_progress(frames):
            frame_results.append(
                dict(boxes=self.multicam_fusion(frame),
                     sample_token=frame[0]['sample_token']))
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self._format_bbox(frame_results, jsonfile_prefix)
        return result_files, tmp_dir

    def _format_bbox(self, results, jsonfile_prefix):
        nusc_annos = {}
        print('Formatting results...')
        for det in mmcv.track_iter_progress(results):
            annos = []
            boxes = det['boxes']
            sample_token = det['sample_token']
            for i, box in enumerate(boxes):
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=self.CLASSES[box.label],
                    detection_score=box.score,
                    attribute_name=box.name)
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def multicam_fusion(self, frame):
        sample_token = frame[0]['sample_token']
        num_classes = len(self.CLASSES)
        boxes_nus_multicls = [[] for _ in range(num_classes)]
        for cam in frame:
            assert cam['sample_token'] == sample_token
            sensor2ego_rotation = Quaternion(cam['sensor2ego_rotation'])
            sensor2ego_translation = np.array(cam['sensor2ego_translation'])
            ego2global_rotation = Quaternion(cam['ego2global_rotation'])
            ego2global_translation = np.array(cam['ego2global_translation'])
            for label_id in range(num_classes):
                bboxes_3d = cam['bbox_3d_results'][label_id]
                for bbox_3d in bboxes_3d:
                    box_nus = self.bbox_3d_to_box_nus(bbox_3d, label_id)
                    # to ego
                    box_nus.rotate(sensor2ego_rotation)
                    box_nus.translate(sensor2ego_translation)
                    # distance filtering
                    cls_range_map = self.eval_detection_configs.class_range
                    radius = np.linalg.norm(box_nus.center[:2], 2)
                    det_range = cls_range_map[self.CLASSES[box_nus.label]]
                    if radius > det_range:
                        continue
                    # to global
                    box_nus.rotate(ego2global_rotation)
                    box_nus.translate(ego2global_translation)
                    boxes_nus_multicls[label_id].append(box_nus)
        boxes_nus = self.multiclass_nms(boxes_nus_multicls)
        if len(boxes_nus) > 500:
            boxes_nus.sort(reverse=True, key=lambda box: box.score)
            boxes_nus = boxes_nus[:500]
        return boxes_nus

    def bbox_3d_to_box_nus(self, bbox_3d, label):
        center = bbox_3d[3:6]
        size = bbox_3d[[2, 0, 1]]  # wlh
        orientation = (
            Quaternion(axis=[0, 1, 0], radians=bbox_3d[6])
            * Quaternion(matrix=self.KITTI2NUS_ROT.T))
        score = bbox_3d[7]
        velocity = (bbox_3d[8], 0, bbox_3d[9])
        attr_score = bbox_3d[10:19]

        cls_name = self.CLASSES[label]
        attr_scope = self.CLS2ATTR[cls_name]
        attr_id_scope = [self.Attributes.index(attr) for attr in attr_scope]
        attr_id = attr_id_scope[attr_score[attr_id_scope].argmax()]
        attr_name = self.Attributes[attr_id]

        return Box(center, size, orientation, label, score, velocity, attr_name)

    def multiclass_nms(self, boxes_nus_multicls, device='cuda:0'):
        boxes_nus_flatten = list_flatten(boxes_nus_multicls)
        xywhr = boxes_nus_to_xywhr(boxes_nus_flatten)
        scores = [box_nus.score for box_nus in boxes_nus_flatten]
        labels = []
        for label, boxes_nus_cls in enumerate(boxes_nus_multicls):
            labels += [label for _ in boxes_nus_cls]
        labels = np.array(labels)
        if len(labels) > 0:
            boxes_for_nms = xywhr2xyxyr(xywhr)
            offset_unit = (boxes_for_nms[:, :4].max() - boxes_for_nms[:, :4].min()) * 2
            boxes_for_nms[:, :4] = boxes_for_nms[:, :4] + (offset_unit * labels)[:, None]
            keep_inds = nms_gpu(
                torch.from_numpy(boxes_for_nms).to(device=device, dtype=torch.float32),
                torch.tensor(scores, device=device, dtype=torch.float32),
                self.nms_thr
            ).cpu().numpy()
        else:
            keep_inds = np.zeros(0, dtype=np.int64)
        boxes_nus_out = [boxes_nus_flatten[ind] for ind in keep_inds]
        return boxes_nus_out


def boxes_nus_to_xywhr(boxes_nus):
    nb = len(boxes_nus)
    xywhr = np.empty((nb, 5), dtype=np.float32)
    for i, box_nus in enumerate(boxes_nus):
        xywhr[i, :2] = box_nus.center[:2]  # [x, y]
        xywhr[i, 2:4] = box_nus.wlh[[1, 0]]  # [l, w]
        rotation_matrix = box_nus.rotation_matrix
        xywhr[i, 4] = np.arctan2(
            rotation_matrix[0, 1] - rotation_matrix[1, 0],
            rotation_matrix[0, 0] + rotation_matrix[1, 1],
            dtype=np.float32)
    return xywhr


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (ndarray): Rotated boxes in XYWHR format.

    Returns:
        ndarray: Converted boxes in XYXYR format.
    """
    boxes = np.zeros_like(boxes_xywhr)
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


def list_flatten(t):
    return [item for sublist in t for item in sublist]
