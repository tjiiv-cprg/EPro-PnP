"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

from __future__ import print_function, division, absolute_import

import math
import os, sys
from six.moves import cPickle
from scipy import spatial
import numpy as np
import ref
from scipy.linalg import logm
import numpy.linalg as LA
import matplotlib.pyplot as plt
import utils.fancy_logger as logger

class Evaluation(object):
    def __init__(self, cfg, models_info, models):
        self.models_info = models_info
        self.models = models
        self.pose_est_all = {}
        self.pose_gt_all = {}
        self.num = {}
        self.numAll = 0.
        self.classes = cfg.classes
        self.camera_matrix = cfg.camera_matrix
        for cls in self.classes:
            self.num[cls] = 0.
            self.pose_est_all[cls] = []
            self.pose_gt_all[cls] = []

    def evaluate_pose(self):
        """
        Evaluate 6D pose and display
        """
        all_poses_est = self.pose_est_all
        all_poses_gt = self.pose_gt_all
        logger.info('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Evaluation 6D Pose', '-' * 100))
        rot_thresh_list = np.arange(1, 11, 1)
        trans_thresh_list = np.arange(0.01, 0.11, 0.01)
        num_metric = len(rot_thresh_list)
        num_classes = len(self.classes)
        rot_acc = np.zeros((num_classes, num_metric))
        trans_acc = np.zeros((num_classes, num_metric))
        space_acc = np.zeros((num_classes, num_metric))

        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            curr_poses_gt = all_poses_gt[cls_name]
            curr_poses_est = all_poses_est[cls_name]
            num = len(curr_poses_gt)
            cur_rot_rst = np.zeros((num, 1))
            cur_trans_rst = np.zeros((num, 1))

            for j in range(num):
                r_dist_est, t_dist_est = calc_rt_dist_m(curr_poses_est[j], curr_poses_gt[j])
                if cls_name == 'eggbox' and r_dist_est > 90:
                    RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                    curr_pose_est_sym = se3_mul(curr_poses_est[j], RT_z)
                    r_dist_est, t_dist_est = calc_rt_dist_m(curr_pose_est_sym, curr_poses_gt[j])
                # logger.info('t_dist: {}'.format(t_dist_est))
                cur_rot_rst[j, 0] = r_dist_est
                cur_trans_rst[j, 0] = t_dist_est

            # cur_rot_rst = np.vstack(all_rot_err[cls_idx, iter_i])
            # cur_trans_rst = np.vstack(all_trans_err[cls_idx, iter_i])
            for thresh_idx in range(num_metric):
                rot_acc[i, thresh_idx] = np.mean(cur_rot_rst < rot_thresh_list[thresh_idx])
                trans_acc[i, thresh_idx] = np.mean(cur_trans_rst < trans_thresh_list[thresh_idx])
                space_acc[i, thresh_idx] = np.mean(np.logical_and(cur_rot_rst < rot_thresh_list[thresh_idx],
                                                                  cur_trans_rst < trans_thresh_list[thresh_idx]))

            logger.info("------------ {} -----------".format(cls_name))
            logger.info("{:>24}: {:>7}, {:>7}, {:>7}".format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"))
            logger.info(
                "{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}".format('average_accuracy', '[{:>2}, {:.2f}]'.format(-1, -1),
                                                                   np.mean(rot_acc[i, :]) * 100,
                                                                   np.mean(trans_acc[i, :]) * 100,
                                                                   np.mean(space_acc[i, :]) * 100))
            show_list = [1, 4, 9]
            for show_idx in show_list:
                logger.info("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                            .format('average_accuracy',
                                    '[{:>2}, {:.2f}]'.format(rot_thresh_list[show_idx], trans_thresh_list[show_idx]),
                                    rot_acc[i, show_idx] * 100, trans_acc[i, show_idx] * 100,
                                    space_acc[i, show_idx] * 100))
        print(' ')
        # overall performance
        show_list = [1, 4, 9]
        logger.info("---------- performance over {} classes -----------".format(num_valid_class))
        logger.info("{:>24}: {:>7}, {:>7}, {:>7}"
                    .format("[rot_thresh, trans_thresh", "RotAcc", "TraAcc", "SpcAcc"))
        logger.info(
            "{:<16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}".format('average_accuracy', '[{:>2}, {:4.2f}]'.format(-1, -1),
                                                               np.sum(rot_acc[:, :]) / (
                                                                           num_valid_class * num_metric) * 100,
                                                               np.sum(trans_acc[:, :]) / (
                                                                       num_valid_class * num_metric) * 100,
                                                               np.sum(space_acc[:, :]) / (
                                                                       num_valid_class * num_metric) * 100))
        for show_idx in show_list:
            logger.info("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}"
                        .format('average_accuracy',
                                '[{:>2}, {:.2f}]'.format(rot_thresh_list[show_idx], trans_thresh_list[show_idx]),
                                np.sum(rot_acc[:, show_idx]) / num_valid_class * 100,
                                np.sum(trans_acc[:, show_idx]) / num_valid_class * 100,
                                np.sum(space_acc[:, show_idx]) / num_valid_class * 100))
        print(' ')

    def evaluate_pose_add(self, output_dir):
        """
        Evaluate 6D pose by ADD Metric
        """
        all_poses_est = self.pose_est_all
        all_poses_gt = self.pose_gt_all
        models_info = self.models_info
        models = self.models
        logger.info('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ADD', '-' * 100))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        eval_method = 'add'
        num_classes = len(self.classes)
        count_all = np.zeros((num_classes), dtype=np.float32)
        count_correct = {k: np.zeros((num_classes), dtype=np.float32) for k in ['0.02', '0.05', '0.10']}

        threshold_002 = np.zeros((num_classes), dtype=np.float32)
        threshold_005 = np.zeros((num_classes), dtype=np.float32)
        threshold_010 = np.zeros((num_classes), dtype=np.float32)
        dx = 0.0001
        threshold_mean = np.tile(np.arange(0, 0.1, dx).astype(np.float32), (num_classes, 1))  # (num_class, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((num_classes, num_thresh), dtype=np.float32)

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            threshold_002[i] = 0.02 * models_info[ref.obj2idx(cls_name)]['diameter']
            threshold_005[i] = 0.05 * models_info[ref.obj2idx(cls_name)]['diameter']
            threshold_010[i] = 0.10 * models_info[ref.obj2idx(cls_name)]['diameter']
            threshold_mean[i, :] *= models_info[ref.obj2idx(cls_name)]['diameter']
            curr_poses_gt = all_poses_gt[cls_name]
            curr_poses_est = all_poses_est[cls_name]
            num = len(curr_poses_gt)
            count_all[i] = num
            for j in range(num):
                RT = curr_poses_est[j]  # est pose
                pose_gt = curr_poses_gt[j]  # gt pose
                if cls_name == 'eggbox' or cls_name == 'glue' or cls_name == 'bowl' or cls_name == 'cup':
                    eval_method = 'adi'
                    error = adi(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3], models[cls_name])
                else:
                    error = add(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3], models[cls_name])
                if error < threshold_002[i]:
                    count_correct['0.02'][i] += 1
                if error < threshold_005[i]:
                    count_correct['0.05'][i] += 1
                if error < threshold_010[i]:
                    count_correct['0.10'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1

        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_002 = np.zeros(1)
        sum_acc_005 = np.zeros(1)
        sum_acc_010 = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            logger.info("** {} **".format(cls_name))
            from scipy.integrate import simps
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_002 = 100 * float(count_correct['0.02'][i]) / float(count_all[i])
            sum_acc_002[0] += acc_002
            acc_005 = 100 * float(count_correct['0.05'][i]) / float(count_all[i])
            sum_acc_005[0] += acc_005
            acc_010 = 100 * float(count_correct['0.10'][i]) / float(count_all[i])
            sum_acc_010[0] += acc_010

            fig = plt.figure()
            x_s = np.arange(0, 0.1, dx).astype(np.float32)
            y_s = count_correct['mean'][i] / float(count_all[i])
            plot_data[cls_name].append((x_s, y_s))
            plt.plot(x_s, y_s, '-')
            plt.xlim(0, 0.1)
            plt.ylim(0, 1)
            plt.xlabel("Average distance threshold in meter (symmetry)")
            plt.ylabel("accuracy")
            plt.savefig(os.path.join(output_dir, 'acc_thres_{}.png'.format(cls_name, )), dpi=fig.dpi)
            plt.close()
            logger.info('threshold=[0.0, 0.10], area: {:.2f}'.format(acc_mean))
            logger.info('threshold=0.02, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.02'][i],
                count_all[i],
                acc_002))
            logger.info('threshold=0.05, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.05'][i],
                count_all[i],
                acc_005))
            logger.info('threshold=0.10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['0.10'][i],
                count_all[i],
                acc_010))
            logger.info(" ")

        with open(os.path.join(output_dir, '{}_xys.pkl'.format(eval_method)), 'wb') as f:
            cPickle.dump(plot_data, f, protocol=2)

        logger.info("=" * 30)
        logger.info(' ')
        # overall performance of add
        for iter_i in range(1):
            logger.info("---------- add performance over {} classes -----------".format(num_valid_class))
            logger.info("** iter {} **".format(iter_i + 1))
            logger.info('threshold=[0.0, 0.10], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            logger.info('threshold=0.02, mean accuracy: {:.2f}'.format(
                sum_acc_002[iter_i] / num_valid_class))
            logger.info('threshold=0.05, mean accuracy: {:.2f}'.format(
                sum_acc_005[iter_i] / num_valid_class))
            logger.info('threshold=0.10, mean accuracy: {:.2f}'.format(
                sum_acc_010[iter_i] / num_valid_class))
            logger.info(' ')
        logger.info("=" * 30)


    def evaluate_pose_arp_2d(self, output_dir):
        '''
        evaluate average re-projection 2d error
        '''
        all_poses_est = self.pose_est_all
        all_poses_gt = self.pose_gt_all
        models = self.models
        logger.info('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Metric ARP_2D (Average Re-Projection 2D)', '-' * 100))
        K = self.camera_matrix
        num_classes = len(self.classes)
        count_all = np.zeros((num_classes), dtype=np.float32)
        count_correct = {k: np.zeros((num_classes), dtype=np.float32) for k in ['2', '5', '10', '20']}

        threshold_2 = np.zeros((num_classes), dtype=np.float32)
        threshold_5 = np.zeros((num_classes), dtype=np.float32)
        threshold_10 = np.zeros((num_classes), dtype=np.float32)
        threshold_20 = np.zeros((num_classes), dtype=np.float32)
        dx = 0.1
        threshold_mean = np.tile(np.arange(0, 50, dx).astype(np.float32),
                                 (num_classes, 1))  # (num_class, num_iter, num_thresh)
        num_thresh = threshold_mean.shape[-1]
        count_correct['mean'] = np.zeros((num_classes, num_thresh), dtype=np.float32)

        for i in range(num_classes):
            threshold_2[i] = 2
            threshold_5[i] = 5
            threshold_10[i] = 10
            threshold_20[i] = 20

        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            curr_poses_gt = all_poses_gt[cls_name]
            curr_poses_est = all_poses_est[cls_name]
            num = len(curr_poses_gt)
            count_all[i] = num
            for j in range(num):
                RT = curr_poses_est[j]  # est pose
                pose_gt = curr_poses_gt[j]  # gt pose
                error_rotation = re(RT[:3, :3], pose_gt[:3, :3])
                if cls_name == 'eggbox' and error_rotation > 90:
                    RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
                    RT_sym = se3_mul(RT, RT_z)
                    error = arp_2d(RT_sym[:3, :3], RT_sym[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                   models[cls_name], K)
                else:
                    error = arp_2d(RT[:3, :3], RT[:, 3], pose_gt[:3, :3], pose_gt[:, 3],
                                   models[cls_name], K)

                if error < threshold_2[i]: count_correct['2'][i] += 1
                if error < threshold_5[i]: count_correct['5'][i] += 1
                if error < threshold_10[i]: count_correct['10'][i] += 1
                if error < threshold_20[i]: count_correct['20'][i] += 1
                for thresh_i in range(num_thresh):
                    if error < threshold_mean[i, thresh_i]:
                        count_correct['mean'][i, thresh_i] += 1

        # store plot data
        plot_data = {}
        sum_acc_mean = np.zeros(1)
        sum_acc_02 = np.zeros(1)
        sum_acc_05 = np.zeros(1)
        sum_acc_10 = np.zeros(1)
        sum_acc_20 = np.zeros(1)
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            plot_data[cls_name] = []
            logger.info("** {} **".format(cls_name))
            from scipy.integrate import simps
            area = simps(count_correct['mean'][i] / float(count_all[i]), dx=dx) / (50.0)
            acc_mean = area * 100
            sum_acc_mean[0] += acc_mean
            acc_02 = 100 * float(count_correct['2'][i]) / float(count_all[i])
            sum_acc_02[0] += acc_02
            acc_05 = 100 * float(count_correct['5'][i]) / float(count_all[i])
            sum_acc_05[0] += acc_05
            acc_10 = 100 * float(count_correct['10'][i]) / float(count_all[i])
            sum_acc_10[0] += acc_10
            acc_20 = 100 * float(count_correct['20'][i]) / float(count_all[i])
            sum_acc_20[0] += acc_20

            fig = plt.figure()
            x_s = np.arange(0, 50, dx).astype(np.float32)
            y_s = 100 * count_correct['mean'][i] / float(count_all[i])
            plot_data[cls_name].append((x_s, y_s))
            plt.plot(x_s, y_s, '-')
            plt.xlim(0, 50)
            plt.ylim(0, 100)
            plt.grid(True)
            plt.xlabel("px")
            plt.ylabel("correctly estimated poses in %")
            plt.savefig(os.path.join(output_dir, 'arp_2d_{}.png'.format(cls_name)), dpi=fig.dpi)
            plt.close()

            logger.info('threshold=[0, 50], area: {:.2f}'.format(acc_mean))
            logger.info('threshold=2, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['2'][i], count_all[i], acc_02))
            logger.info('threshold=5, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                count_correct['5'][i], count_all[i], acc_05))
            logger.info(
                'threshold=10, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['10'][i], count_all[i], acc_10))
            logger.info(
                'threshold=20, correct poses: {}, all poses: {}, accuracy: {:.2f}'.format(
                    count_correct['20'][i], count_all[i], acc_20))
            logger.info(" ")

        with open(os.path.join(output_dir, 'arp_2d_xys.pkl'), 'wb') as f:
            cPickle.dump(plot_data, f, protocol=2)
        logger.info("=" * 30)
        logger.info(' ')
        # overall performance of arp 2d
        for iter_i in range(1):
            logger.info("---------- arp 2d performance over {} classes -----------".format(num_valid_class))
            logger.info("** iter {} **".format(iter_i + 1))
            logger.info('threshold=[0, 50], area: {:.2f}'.format(
                sum_acc_mean[iter_i] / num_valid_class))
            logger.info('threshold=2, mean accuracy: {:.2f}'.format(
                sum_acc_02[iter_i] / num_valid_class))
            logger.info('threshold=5, mean accuracy: {:.2f}'.format(
                sum_acc_05[iter_i] / num_valid_class))
            logger.info('threshold=10, mean accuracy: {:.2f}'.format(
                sum_acc_10[iter_i] / num_valid_class))
            logger.info('threshold=20, mean accuracy: {:.2f}'.format(
                sum_acc_20[iter_i] / num_valid_class))
            logger.info(" ")
        logger.info("=" * 30)
        

    def evaluate_trans(self):
        '''
        evaluate trans error in detail
        '''
        all_poses_est = copy.deepcopy(self.pose_est_all)
        all_poses_gt = copy.deepcopy(self.pose_gt_all)

        logger.info('\n* {} *\n {:^}\n* {} *'.format('-' * 100, 'Evaluation Translation', '-' * 100))
        rot_thresh_list = np.arange(1, 11, 1)
        trans_thresh_list = np.arange(0.01, 0.11, 0.01)
        num_metric = len(rot_thresh_list)
        num_classes = len(self.classes)

        trans_acc = np.zeros((num_classes, num_metric))
        x_acc = np.zeros((num_classes, num_metric))
        y_acc = np.zeros((num_classes, num_metric))
        z_acc = np.zeros((num_classes, num_metric))

        num_classes = len(self.classes)

        threshold_2 = np.zeros((num_classes, 3), dtype=np.float32)
        threshold_5 = np.zeros((num_classes, 3), dtype=np.float32)
        threshold_10 = np.zeros((num_classes, 3), dtype=np.float32)
        threshold_20 = np.zeros((num_classes, 3), dtype=np.float32)

        for i in range(num_classes):
            for j in range(3):
                threshold_2[i][j] = 2
                threshold_5[i][j] = 5
                threshold_10[i][j] = 10
                threshold_20[i][j] = 20

        num_valid_class = len(self.classes)
        for i, cls_name in enumerate(self.classes):
            curr_poses_gt = all_poses_gt[cls_name]
            curr_poses_est = all_poses_est[cls_name]
            num = len(curr_poses_gt)
            cur_trans_rst = np.zeros((num, 1))
            cur_x_rst = np.zeros((num, 1))
            cur_y_rst = np.zeros((num, 1))
            cur_z_rst = np.zeros((num, 1))

            for j in range(num):
                RT = curr_poses_est[j]  # est pose
                pose_gt = curr_poses_gt[j]  # gt pose
                t_dist_est = LA.norm(RT[:, 3].reshape(3) - pose_gt[:, 3].reshape(3))
                err_xyz = np.abs(RT[:, 3] - pose_gt[:, 3])
                cur_x_rst[j, 0], cur_y_rst[j, 0], cur_z_rst[j, 0] = err_xyz
                cur_trans_rst[j, 0] = t_dist_est

            for thresh_idx in range(num_metric):
                trans_acc[i, thresh_idx] = np.mean(cur_trans_rst < trans_thresh_list[thresh_idx])
                x_acc[i, thresh_idx] = np.mean(cur_x_rst < trans_thresh_list[thresh_idx])
                y_acc[i, thresh_idx] = np.mean(cur_y_rst < trans_thresh_list[thresh_idx])
                z_acc[i, thresh_idx] = np.mean(cur_z_rst < trans_thresh_list[thresh_idx])

            logger.info("------------ {} -----------".format(cls_name))
            logger.info("{:>24}: {:>7}, {:>7}, {:>7}, {:>7}".format("trans_thresh", "TraAcc", "x", "y", "z"))
            show_list = [1, 4, 9]
            for show_idx in show_list:
                logger.info("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}".format('average_accuracy',
                                    '{:.2f}'.format(trans_thresh_list[show_idx]),
                                    trans_acc[i, show_idx] * 100, x_acc[i, show_idx] * 100,
                                    y_acc[i, show_idx] * 100, z_acc[i, show_idx] * 100))
        print(' ')
        # overall performance
        show_list = [1, 4, 9]
        logger.info("---------- performance over {} classes -----------".format(num_valid_class))
        logger.info("{:>24}: {:>7}, {:>7}, {:>7}, {:>7}".format("trans_thresh", "TraAcc", "x", "y", "z"))

        for show_idx in show_list:
            logger.info("{:>16}{:>8}: {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}".format('average_accuracy',
                                '{:.2f}'.format(trans_thresh_list[show_idx]),
                                np.sum(trans_acc[:, show_idx]) / num_valid_class * 100,
                                np.sum(x_acc[:, show_idx]) / num_valid_class * 100,
                                np.sum(y_acc[:, show_idx]) / num_valid_class * 100,
                                np.sum(z_acc[:, show_idx]) / num_valid_class * 100))
        print(' ')

    
def AccCls(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


sqrt2 = 2.0 ** 0.5
pi = np.arccos(-1)
'''
    % First roation along Z by azimuth. Then along X by -(pi/2-elevation).
    % Then along Z by theta
'''


def RotMat(axis, ang):
    s = np.sin(ang)
    c = np.cos(ang)
    res = np.zeros((3, 3))
    if axis == 'Z':
        res[0, 0] = c
        res[0, 1] = -s
        res[1, 0] = s
        res[1, 1] = c
        res[2, 2] = 1
    elif axis == 'Y':
        res[0, 0] = c
        res[0, 2] = s
        res[1, 1] = 1
        res[2, 0] = -s
        res[2, 2] = c
    elif axis == 'X':
        res[0, 0] = 1
        res[1, 1] = c
        res[1, 2] = -s
        res[2, 1] = s
        res[2, 2] = c
    return res


def angle2dcm(angle):
    azimuth = angle[0]
    elevation = angle[1]
    theta = angle[2]
    return np.dot(RotMat('Z', theta), np.dot(RotMat('X', - (pi / 2 - elevation)), RotMat('Z', - azimuth)))


def AccViewCls(output, target, numBins, specificView):
    # unified
    binSize = 360. / numBins
    if specificView:
        acc = 0
        for t in range(target.shape[0]):
            idx = np.where(target[t] != numBins)
            ps = idx[0][0] / 3 * 3
            _, pred = output[t].view(-1, numBins)[ps: ps + 3].topk(1, 1, True, True)
            pred = pred.view(3).float() * binSize / 180. * pi
            gt = target[t][ps: ps + 3].float() * binSize / 180. * pi
            R_pred = angle2dcm(pred)
            R_gt = angle2dcm(gt)
            err = ((logm(np.dot(np.transpose(R_pred), R_gt)) ** 2).sum()) ** 0.5 / sqrt2
            acc += 1 if err < pi / 6. else 0
        return 1.0 * acc / target.shape[0]
    else:
        _, pred = output.view(target.shape[0] * 3, numBins).topk(1, 1, True, True)
        pred = pred.view(target.shape[0], 3).float() * binSize / 180. * pi
        target = target.float() * binSize / 180. * pi
        acc = 0
        for t in range(target.shape[0]):
            R_pred = angle2dcm(pred[t])
            R_gt = angle2dcm(target[t])
            err = ((logm(np.dot(np.transpose(R_pred), R_gt)) ** 2).sum()) ** 0.5 / sqrt2
            acc += 1 if err < pi / 6. else 0
        return 1.0 * acc / target.shape[0]


# ------------------------------------------------------------------------------------ #
# lzg adds
# ------------------------------------------------------------------------------------ #
def se3_mul(RT1, RT2):
    """
    concat 2 RT transform
    :param RT1=[R,T], 4x3 np array
    :param RT2=[R,T], 4x3 np array
    :return: RT_new = RT1 * RT2
    """
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert (pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T


def transform_pts_Rt_2d(pts, R, t, K):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :param K: 3x3 intrinsic matrix
    :return: nx2 ndarray with transformed 2D points.
    """
    assert (pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))  # 3xn
    pts_c_t = K.dot(pts_t)
    n = pts.shape[0]
    pts_2d = np.zeros((n, 2))
    pts_2d[:, 0] = pts_c_t[0, :] / pts_c_t[2, :]
    pts_2d[:, 1] = pts_c_t[1, :] / pts_c_t[2, :]

    return pts_2d


def arp_2d(R_est, t_est, R_gt, t_gt, pts, K):
    '''
    average re-projection error in 2d

    :param R_est:
    :param t_est:
    :param R_gt:
    :param t_gt:
    :param pts:
    :param K:
    :return:
    '''
    pts_est_2d = transform_pts_Rt_2d(pts, R_est, t_est, K)
    pts_gt_2d = transform_pts_Rt_2d(pts, R_gt, t_gt, K)
    e = np.linalg.norm(pts_est_2d - pts_gt_2d, axis=1).mean()
    return e


def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e


def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e


def re_old(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert (R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos))  # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi  # [rad] -> [deg]
    return error

def re(R_est, R_gt):
    assert (R_est.shape == R_gt.shape == (3, 3))
    temp = logm(np.dot(np.transpose(R_est), R_gt))
    rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180
    return rd_deg


def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert (t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt.reshape(3) - t_est.reshape(3))
    return error


# def calc_rt_dist_m_v1(pose_src, pose_tgt):
#     R_src = pose_src[:, :3]
#     T_src = pose_src[:, 3]
#     R_tgt = pose_tgt[:, :3]
#     T_tgt = pose_tgt[:, 3]
#     error_cos = 0.5 * (np.trace(R_src.dot(np.linalg.inv(R_tgt))) - 1.0)
#     error_cos = min(1.0, max(-1.0, error_cos))  # Avoid invalid values due to numerical errors
#     rd_rad = math.acos(error_cos)
#     rd_deg = 180.0 * rd_rad / np.pi  # [rad] -> [deg]
#
#     td = LA.norm(T_tgt - T_src)
#
#     return rd_deg, td


def calc_rt_dist_m(pose_src, pose_tgt):
    R_src = pose_src[:, :3]
    T_src = pose_src[:, 3]
    R_tgt = pose_tgt[:, :3]
    T_tgt = pose_tgt[:, 3]
    temp = logm(np.dot(np.transpose(R_src), R_tgt))
    rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
    rd_deg = rd_rad / np.pi * 180

    td = LA.norm(T_tgt - T_src)

    return rd_deg, td


def calc_all_errs(R_est, T_est, R_gt, T_gt, model_points, K, cls_name):
    """
    Calculate all pose errors.

    :param R_est, T_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, T_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param pts: Object model (nx3 ndarray with 3D model points)
    :param K:
    :return: Errors of pose_est w.r.t. pose_gt
    """
    RT_est = np.concatenate((R_est, T_est.reshape(3, 1)), 1)
    RT = np.concatenate((R_gt, T_gt.reshape(3, 1)), 1)

    err_R = re(RT_est[:, :3], RT[:, :3])
    if cls_name == 'eggbox' and err_R > 90:
        RT_z = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0]])
        RT_est_sym = se3_mul(RT_est, RT_z)
        err_R, err_T = calc_rt_dist_m(RT_est_sym, RT)
        ARP_2D = arp_2d(RT_est_sym[:, :3], RT_est_sym[:, 3], R_gt, T_gt, model_points, K)

    else:
        err_R, err_T = calc_rt_dist_m(RT_est, RT)
        ARP_2D = arp_2d(R_est, T_est, R_gt, T_gt, model_points, K)

    if cls_name in ['eggbox', 'glue']:
        ADD_or_ADI = adi(R_est, T_est, R_gt, T_gt, model_points)
    else:
        ADD_or_ADI = add(R_est, T_est, R_gt, T_gt, model_points)

    return err_R, err_T, ARP_2D, ADD_or_ADI




