"""
This file is modified from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import os
import yaml
import argparse
import copy
import numpy as np
from easydict import EasyDict as edict
import sys
import ref
from datetime import datetime
import utils.fancy_logger as logger
from tensorboardX import SummaryWriter
from pprint import pprint


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_default_config_pytorch():
    config = edict()
    config.exp_id = 'CDPN'          # Experiment ID
    config.task = 'rot'             # 'rot | trans | trans_rot'
    config.gpu = 1
    config.threads_num = 12         # 'nThreads'
    config.debug = False
    config.demo = '../demo'         # demo save path
    config.debug_path = '../debug'  # debug save path
    config.save_mode = 'all'        # 'all' | 'best', save all models or only save the best model
    config.load_model = ''          # path to a previously trained model
    config.test = False             # run in test mode or not
    return config

def get_default_dataset_config():
    config = edict()
    config.name = 'LineMOD_No_Occ'
    config.classes = 'all'
    config.img_type = 'imgn'            # 'real' | 'imgn' | 'real_imgn'
    config.syn_num = 1000
    config.syn_samp_type = 'uniform'  # 'uniform' | 'random'
    return config

def get_default_dataiter_config():
    config = edict()
    config.inp_res = 256
    config.out_res = 64
    config.dzi = True
    config.denoise_coor = True
    return config

def get_default_augment_config():
    config = edict()
    config.change_bg_ratio = 0.5
    config.pad_ratio = 1.5
    config.scale_ratio = 0.25
    config.shift_ratio = 0.25
    return config

def get_default_train_config():
    config = edict()
    config.begin_epoch = 1
    config.end_epoch = 160
    config.test_interval = 10
    config.train_batch_size = 6
    config.lr_backbone = 1e-4
    config.lr_rot_head = 1e-4
    config.lr_trans_head = 1e-4
    config.lr_epoch_step = [50, 100, 150]
    config.lr_factor = 0.1
    config.optimizer_name = 'RMSProp'  # 'Adam' | 'Sgd' | 'Moment' | 'RMSProp'
    config.warmup_lr = 1e-5
    config.warmup_step = 500
    config.momentum = 0.0
    config.weightDecay = 0.0
    config.alpha = 0.99
    config.epsilon = 1e-8
    return config

def get_default_loss_config():
    config = edict()
    config.rot_loss_type = 'L1'
    config.rot_mask_loss = True
    config.rot_loss_weight = 1
    config.trans_loss_type = 'L2'
    config.trans_loss_weight = 1
    config.mc_loss_weight = 0.02
    config.t_loss_weight = 0.
    config.r_loss_weight = 0.
    return config

def get_default_network_config():
    config = edict()
    # ------ backbone -------- #
    config.arch = 'resnet'
    config.back_freeze = False
    config.back_input_channel = 3 # # channels of backbone's input
    config.back_layers_num = 34   # 18 | 34 | 50 | 101 | 152
    config.back_filters_num = 256  # number of filters for each layer
    # ------ rotation head -------- #
    config.rot_head_freeze = False
    config.rot_layers_num = 3
    config.rot_filters_num = 256  # number of filters for each layer
    config.rot_conv_kernel_size = 3  # kernel size for hidden layers
    config.rot_output_conv_kernel_size = 1  # kernel size for output layer
    config.rot_output_channels = 4  # # channels of output, 3-channels coordinates map and 1-channel for confidence map
    # ------ translation head -------- #
    config.trans_head_freeze = False
    config.trans_layers_num = 3
    config.trans_filters_num = 256
    config.trans_conv_kernel_size = 3
    config.trans_output_channels = 3
    return config

def get_default_test_config():
    config = edict()
    config.test_mode = 'pose'   # 'pose' | 'add' | 'proj' | 'all' | 'pose_fast' | 'add_fast' | 'proj_fast' | 'all_fast'
                                # 'pose' means "#cm, #degrees", 'all' means evaluate on all metrics,
                                # 'fast' means the test batch size equals training batch size, otherwise 1
    config.cache_file = ''
    config.ignore_cache_file = True
    config.erode_mask = False
    config.pnp = 'ransac'  # 'ransac' | 'ePnP' | 'iterpnp'
    config.mask_threshold = 0.5
    config.detection = 'YOLOv3'  # 'YOLOv3' | 'TinyYOLOv3' | 'FasterRCNN'
    config.disp_interval = '200'
    config.vis_demo = False
    # setting for ransac
    config.ransac_projErr = 3
    config.ransac_iterCount = 100
    return config

def get_base_config():
    base_config = edict()
    base_config.pytorch = get_default_config_pytorch()
    base_config.dataset = get_default_dataset_config()
    base_config.dataiter = get_default_dataiter_config()
    base_config.train = get_default_train_config()
    base_config.test = get_default_test_config()
    base_config.augment = get_default_augment_config()
    base_config.network = get_default_network_config()
    base_config.loss = get_default_loss_config()
    return base_config

def update_config_from_file(_config, config_file, check_necessity=True):
    config = copy.deepcopy(_config)
    with open(config_file) as f:
        # exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            if isinstance(vv, list) and not isinstance(vv[0], str):
                                config[k][vk] = np.asarray(vv)
                            else:
                                config[k][vk] = vv
                        else:
                            if check_necessity:
                                raise ValueError("{}.{} not exist in config".format(k, vk))
                else:
                    raise ValueError("{} is not dict type".format(v))
            else:
                if check_necessity:
                    raise ValueError("{} not exist in config".format(k))
    return config

class config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='pose experiment')
        self.parser.add_argument('--cfg', required=True, type=str, help='path/to/configure_file')
        self.parser.add_argument('--load_model', type=str, help='path/to/model, requird when resume/test')
        self.parser.add_argument('--debug', action='store_true', help='')
        self.parser.add_argument('--test', action='store_true', help='')

    def parse(self):
        config = get_base_config()                  # get default arguments
        args, rest = self.parser.parse_known_args() # get arguments from command line
        for k, v in vars(args).items():
            config.pytorch[k] = v 
        config_file = config.pytorch.cfg
        config = update_config_from_file(config, config_file, check_necessity=False) # update arguments from config file
        # complement config regarding dataset
        if config.dataset.name.lower() in ['lm', 'lmo']:
            config.dataset['camera_matrix'] = ref.K
            config.dataset['width'] = 640
            config.dataset['height'] = 480
            if isinstance(config.dataset.classes, str):
                if config.dataset.classes.lower() == 'all':
                    if config.dataset.name.lower() == 'lm':
                        config.dataset.classes = ref.lm_obj
                    if config.dataset.name.lower() == 'lmo':
                        config.dataset.classes = ref.lmo_obj
        else:
            raise Exception("Wrong dataset name: {}".format(config.dataset.name))

        config.dataset['center'] = (config.dataset['height'] / 2, config.dataset['width'] / 2)

        # automatically correct config
        if config.network.back_freeze == True:
            config.loss.backbone_loss_weight = 0
        if config.network.rot_head_freeze == True:
            config.loss.rot_loss_weight = 0
            config.loss.mc_loss_weight = 0
            config.loss.t_loss_weight = 0
            config.loss.r_loss_weight = 0
        if config.network.trans_head_freeze == True:
            config.loss.trans_loss_weight = 0

        if config.pytorch.test:
            config.pytorch.exp_id = config.pytorch.exp_id + 'TEST'

        # complement config regarding paths
        now = datetime.now().isoformat()
        # save path
        config.pytorch['save_path'] = os.path.join(ref.exp_dir, config.pytorch.exp_id, now)
        if not os.path.exists(config.pytorch.save_path):
            os.makedirs(config.pytorch.save_path, exist_ok=True)
        # debug path
        if config.pytorch.debug:
            config.pytorch.threads_num = 1
            config.pytorch['debug_path'] = os.path.join(config.pytorch.save_path, 'debug')
            if not os.path.exists(config.pytorch.debug_path):
                os.makedirs(config.pytorch.debug_path)
        # demo path
        if config.pytorch.demo:
            config.pytorch['demo_path'] = os.path.join(config.pytorch.save_path, 'demo')
            if not os.path.exists(config.pytorch.demo_path):
                os.makedirs(config.pytorch.demo_path)
        # tensorboard path
        config.pytorch['tensorboard'] = os.path.join(config.pytorch.save_path, 'tensorboard')
        if not os.path.exists(config.pytorch.tensorboard):
            os.makedirs(config.pytorch.tensorboard)
        config.writer = SummaryWriter(config.pytorch.tensorboard)
        # logger path
        logger.set_logger_dir(config.pytorch.save_path, action='k')

        pprint(config)
        # copy and save current config file
        os.system('cp {} {}'.format(config_file, os.path.join(config.pytorch.save_path, 'config_copy.yaml')))
        # save all config infos
        args = dict((name, getattr(config, name)) for name in dir(config) if not name.startswith('_'))
        refs = dict((name, getattr(ref, name)) for name in dir(ref) if not name.startswith('_'))
        file_name = os.path.join(config.pytorch.save_path, 'config.txt')
        with open(file_name, 'wt') as cfg_file:
            cfg_file.write('==> Cmd:\n')
            cfg_file.write(str(sys.argv))
            cfg_file.write('\n==> Opt:\n')
            for k, v in sorted(args.items()):
                cfg_file.write('  %s: %s\n' % (str(k), str(v)))
            cfg_file.write('==> Ref:\n')
            for k, v in sorted(refs.items()):
                cfg_file.write('  %s: %s\n' % (str(k), str(v)))

        return config
