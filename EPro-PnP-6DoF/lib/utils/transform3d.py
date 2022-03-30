"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import numpy as np
import torch 

def prj_vtx_cam(vtx, cam_K):
    """
    project 3D vertices to 2-dimensional image plane
    :param vtx: (N, 3) or (3, ), vertices
    :param cam_K: (3, 3), intrinsic camera parameter
    :return: pts_2D: (N, 2), pixel coordinates; z: (N,), depth
    """
    sp = vtx.shape
    if (len(sp) == 1) and (sp[0] == 3):
        single = True
    elif (len(sp) == 2) and (sp[1] == 3):
        single = False
    else:
        raise
    if single:
        vtx = np.asarray(vtx)[None, :] 
    pts_3d_c = np.matmul(cam_K, vtx.T) 
    pts_2d = pts_3d_c[:2] / pts_3d_c[2]
    z = pts_3d_c[2]
    if single:
        return pts_2d.squeeze(), z
    else:
        return pts_2d.T, z

def prj_vtx_pose(vtx, pose, cam_K):
    """
    project 3D vertices to 2-dimensional image plane by pose
    :param vtx: (N, 3), vertices
    :param pose: (3, 4)
    :param cam_K: (3, 3), intrinsic camera parameter
    :return: pts_2D: (N, 2), pixel coordinates; z: (N,), depth
    """
    pts_3d_w = np.matmul(pose[:, :3], vtx.T) + pose[:, 3].reshape((3, 1)) # (3, N)
    pts_3d_c = np.matmul(cam_K, pts_3d_w) 
    pts_2d = pts_3d_c[:2] / pts_3d_c[2]
    z = pts_3d_w[2]
    return pts_2d.T, z

def prj_vtx_pth(vtx, pose, cam_K):
    """
    project 3D vertices to 2-dimensional image plane using pytorch
    :param vtx: (N, 3), vertices, tensor
    :param pose: (3, 4), tensor
    :param cam_K: (3, 3), intrinsic camera parameter, tensor
    :return: pts_2D: (N, 2), pixel coordinates, tensor; z: (N,), depth, tensor
    """
    pts_3d_w = torch.mm(pose[:, :3], vtx.t) + pose[:, 3].reshape((3, 1)) # (3, N)
    pts_3d_c = torch.mm(cam_K, pts_3d_w) 
    pts_2d = pts_3d_c[:2] / pts_3d_c[2]
    z = pts_3d_w[2]
    return pts_2d.t, z
