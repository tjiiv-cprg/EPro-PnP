"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/open-mmlab/mmdetection3d
"""

import argparse
import mmcv
import numpy as np
import os
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

NameMapping = {
    'movable_object.barrier': 'barrier',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.car': 'car',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.motorcycle': 'motorcycle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'movable_object.trafficcone': 'traffic_cone',
    'vehicle.trailer': 'trailer',
    'vehicle.truck': 'truck'
}


def create_nuscenes_annotations(root_path, info_prefix, version='v1.0-trainval'):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))

    oc_maps_dir = osp.join(root_path, 'oc_maps')
    if not osp.exists(oc_maps_dir):
        os.mkdir(oc_maps_dir)
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, oc_maps_dir)

    metadata = dict(version=version)

    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_annotations_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_annotations_train.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_annotations_val.pkl'.format(info_prefix))
        mmcv.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == '':
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes


def _fill_trainval_infos(nusc, train_scenes, val_scenes, oc_maps_dir):
    train_nusc_infos = []
    val_nusc_infos = []

    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        mmcv.check_file_exist(lidar_path)

        with open(lidar_path, mode='rb') as f:
            lidar_points = f.read()
            f.close()
        lidar_points = np.frombuffer(
            lidar_points, dtype=np.float32).reshape(-1, 5)[:, :3]

        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            ann_records, cam_intrinsic, imsize = get_ann_records(
                nusc,
                cam_token,
                visibilities=['', '1', '2', '3', '4'])
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(
                cam_intrinsic=cam_intrinsic,
                imsize=imsize,
                ann_records=ann_records)
            oc_path = get_obj_crd(lidar_points, cam_info, oc_maps_dir)
            cam_info.update(oc_path=oc_path)
            info['cams'].update({cam: cam_info})

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def get_ann_records(nusc, sample_data_token: str, visibilities: List[str]):
    """Get the 2D and 3D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token: Sample data token belonging to a camera keyframe.
        visibilities: Visibility filter.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])
    imsize = (sd_rec['width'], sd_rec['height'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    ann_recs_converted = []

    for ann_rec in ann_recs:
        if ann_rec['category_name'] not in NameMapping:
            continue

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Project 3D box to image
        final_coords = get_2d_bbox_from_3d(box, camera_intrinsic, imsize)
        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue

        # compute 3d box truncation
        volumn_points_3d = get_volumn_points_3d(box)
        num_total_points = volumn_points_3d.shape[1]
        in_front = volumn_points_3d[2, :] > 0
        volumn_points_coords = view_points(
            volumn_points_3d[:, in_front], camera_intrinsic, True)[:2]
        in_image = ((volumn_points_coords > 0) & (
                volumn_points_coords < np.array(
                    imsize, dtype=volumn_points_coords.dtype)[:, None])).min(axis=0)
        truncation = 1 - float(in_image.sum() / num_total_points)

        # velocity
        global_velo2d = nusc.box_velocity(box.token)[:2]
        global_velo3d = np.array([*global_velo2d, 0.0])
        e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
        c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
        cam_velo3d = global_velo3d @ np.linalg.inv(
            e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
        velo = cam_velo3d[0::2].tolist()

        # Generate dictionary record to be included in the .json file.
        cat_name = NameMapping[ann_rec['category_name']]
        cat_id = nus_categories.index(cat_name)
        ann_token = nusc.get('sample_annotation', box.token)['attribute_tokens']
        if len(ann_token) == 0:
            attr_name = 'None'
        else:
            attr_name = nusc.get('attribute', ann_token[0])['name']
        attr_id = nus_attributes.index(attr_name)
        ann_rec_converted = dict(
            bbox=final_coords,
            bbox3d=box,
            cat_name=cat_name,
            cat_id=cat_id,
            attr_id=attr_id,
            velo=velo,
            instance_token=ann_rec['instance_token'],
            visibility=ann_rec['visibility_token'],
            truncation=truncation,
            num_lidar_pts=ann_rec['num_lidar_pts'],
            num_radar_pts=ann_rec['num_radar_pts'],
        )
        ann_recs_converted.append(ann_rec_converted)

    return ann_recs_converted, camera_intrinsic, imsize


def get_volumn_points_3d(nusc_box, num=10):
    w, l, h = nusc_box.wlh
    x = l * np.linspace(-0.5, 0.5, num=num, dtype=np.float32)
    y = w * np.linspace(-0.5, 0.5, num=num, dtype=np.float32)
    z = h * np.linspace(-0.5, 0.5, num=num, dtype=np.float32)
    points = np.concatenate(np.meshgrid(x, y, z), axis=0).reshape(3, -1)
    # Rotate
    points = np.dot(nusc_box.orientation.rotation_matrix, points)
    # Translate
    points += nusc_box.center[:, None]
    return points


def get_2d_bbox_from_3d(bbox3d, cam_intrinsic, imsize, z_clip=0.1):
    corners, edge_idx = compute_box_3d(bbox3d)
    corners_in_front = corners[:, 2] >= z_clip
    pts_3d = corners[corners_in_front]
    # compute intersection
    edges_0_in_front = corners_in_front[edge_idx[:, 0]]
    edges_1_in_front = corners_in_front[edge_idx[:, 1]]
    edges_clipped = edges_0_in_front ^ edges_1_in_front
    if np.any(edges_clipped):
        edge_idx_to_clip = edge_idx[edges_clipped]
        edges_0 = corners[edge_idx_to_clip[:, 0]]
        edges_1 = corners[edge_idx_to_clip[:, 1]]
        z0 = edges_0[:, 2]
        z1 = edges_1[:, 2]
        weight_0 = z1 - z_clip
        weight_1 = z_clip - z0
        intersection = (edges_0 * weight_0[:, None] + edges_1 * weight_1[:, None]
                        ) * (1 / (z1 - z0)).clip(min=-1e6, max=1e6)[:, None]
        pts_3d = np.concatenate([pts_3d, intersection], axis=0)
    pts_2d = proj_to_img(pts_3d, cam_intrinsic, z_clip=z_clip)
    return post_process_coords(pts_2d, imsize=imsize)


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def get_obj_crd(lidar_points, cam_info, save_dir):
    imsize = cam_info['imsize']

    # get image projection
    cam_points = (lidar_points - cam_info['sensor2lidar_translation']
                  ) @ cam_info['sensor2lidar_rotation']
    in_front = cam_points[:, 2] > 0.1
    cam_points = cam_points[in_front]  # (N, 3)
    proj_cam_points = cam_points @ cam_info['cam_intrinsic'].T  # (N, 3) in [uz, vz, z]
    proj_cam_uv = proj_cam_points[:, :2] / proj_cam_points[:, 2:]  # (N, 2)
    in_canvas = ((proj_cam_uv >= -0.5) & (
            proj_cam_uv < np.array(imsize, dtype=proj_cam_uv.dtype) - 0.5
    )).min(axis=1)
    cam_points = cam_points[in_canvas]  # (n, 3)
    proj_cam_uv = proj_cam_uv[in_canvas]  # (n, 2)

    # get OC
    oc_list = []
    uv_list = []
    for i, ann_record in enumerate(cam_info['ann_records']):
        bbox3d = ann_record['bbox3d']
        oc = (cam_points - bbox3d.center) @ bbox3d.rotation_matrix  # (n, 3)
        w, l, h = bbox3d.wlh
        ub = np.array([l / 2, w / 2, h / 2])
        lb = -ub
        mask = ((oc >= lb) & (oc <= ub)).min(axis=1)  # (n, )
        oc_list.append(oc[mask])
        uv_list.append(proj_cam_uv[mask])

    # save data
    filename = osp.splitext(osp.basename(cam_info['data_path']))[0]
    data_path = osp.join(save_dir, filename) + '__OC.pkl'
    mmcv.dump(dict(oc_list=oc_list, uv_list=uv_list), data_path)

    return data_path


NUS_KITTI_ROT = np.array([[1, 0, 0],
                          [0, 0, 1],
                          [0, -1, 0]])


def proj_to_img(pts, proj_mat, z_clip=1e-4):
    pts_2d = pts @ proj_mat.T
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:].clip(min=z_clip)
    return pts_2d


def compute_box_3d(bbox3d):
    edge_idx = np.array([[0, 1],
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
    bw, bl, bh = bbox3d.wlh
    bottem_center = bbox3d.rotation_matrix @ np.array(
        [0, 0, -0.5 * bh], dtype=np.float64) + bbox3d.center
    rotation_matrix = bbox3d.rotation_matrix @ NUS_KITTI_ROT
    corners = np.array([[ 0.5,  0,  0.5],
                        [ 0.5,  0, -0.5],
                        [-0.5,  0, -0.5],
                        [-0.5,  0,  0.5],
                        [ 0.5, -1,  0.5],
                        [ 0.5, -1, -0.5],
                        [-0.5, -1, -0.5],
                        [-0.5, -1,  0.5]])
    corners *= [bl, bh, bw]
    corners = corners @ rotation_matrix.T + bottem_center
    return corners, edge_idx


def parse_args():
    parser = argparse.ArgumentParser(description='Convert nuScenes dataset')
    parser.add_argument('path', help='nuScenes directory path')
    parser.add_argument('--version',
                        choices=['v1.0-trainval', 'v1.0-test', 'v1.0-mini'],
                        default='v1.0-trainval',
                        help='data version')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_nuscenes_annotations(args.path, 'nuscenes', version=args.version)
