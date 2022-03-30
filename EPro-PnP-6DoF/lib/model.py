"""
This file is from
https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi
"""

import torch
from torch.utils import model_zoo
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck
import os, sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.join(cur_dir, '../..'))
import utils.fancy_logger as logger
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

from models.resnet_backbone import ResNetBackboneNet
from models.resnet_rot_head import RotHeadNet
from models.resnet_trans_head import TransHeadNet
from models.CDPN import CDPN


# Specification
resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}


# Re-init optimizer
def build_model(cfg):
    ## get model and optimizer
    if 'resnet' in cfg.network.arch:
        params_lr_list = []
        # backbone net
        block_type, layers, channels, name = resnet_spec[cfg.network.back_layers_num]
        backbone_net = ResNetBackboneNet(block_type, layers, cfg.network.back_input_channel, cfg.network.back_freeze)
        if cfg.network.back_freeze:
            for param in backbone_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, backbone_net.parameters()),
                                   'lr': float(cfg.train.lr_backbone)})
        # rotation head net
        rot_head_net = RotHeadNet(channels[-1], cfg.network.rot_layers_num, cfg.network.rot_filters_num, cfg.network.rot_conv_kernel_size,
                                  cfg.network.rot_output_conv_kernel_size, cfg.network.rot_output_channels, cfg.network.rot_head_freeze)
        if cfg.network.rot_head_freeze:
            for param in rot_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, rot_head_net.parameters()),
                                   'lr': float(cfg.train.lr_rot_head)})
        # translation head net
        trans_head_net = TransHeadNet(channels[-1], cfg.network.trans_layers_num, cfg.network.trans_filters_num, cfg.network.trans_conv_kernel_size,
                                      cfg.network.trans_output_channels, cfg.network.trans_head_freeze)
        if cfg.network.trans_head_freeze:
            for param in trans_head_net.parameters():
                with torch.no_grad():
                    param.requires_grad = False
        else:
            params_lr_list.append({'params': filter(lambda p: p.requires_grad, trans_head_net.parameters()),
                                   'lr': float(cfg.train.lr_trans_head)})
        # CDPN (Coordinates-based Disentangled Pose Network)
        model = CDPN(backbone_net, rot_head_net, trans_head_net)
        # get optimizer
        if params_lr_list != []:
            optimizer = torch.optim.RMSprop(params_lr_list, alpha=cfg.train.alpha, eps=float(cfg.train.epsilon),
                                            weight_decay=cfg.train.weightDecay, momentum=cfg.train.momentum)
        else:
            optimizer = None

    ## model initialization
    if cfg.pytorch.load_model != '':
        logger.info("=> loading model '{}'".format(cfg.pytorch.load_model))
        checkpoint = torch.load(cfg.pytorch.load_model, map_location=lambda storage, loc: storage)
        if type(checkpoint) == type({}):
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint.state_dict()

        if 'resnet' in cfg.network.arch:
            model_dict = model.state_dict()
            # filter out unnecessary params
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # update state dict
            model_dict.update(filtered_state_dict)
            # load params to net
            model.load_state_dict(model_dict)
    else:
        if 'resnet' in cfg.network.arch:
            logger.info("=> loading official model from model zoo for backbone")
            _, _, _, name = resnet_spec[cfg.network.back_layers_num]
            official_resnet = model_zoo.load_url(model_urls[name])
            # drop original resnet fc layer, add 'None' in case of no fc layer, that will raise error
            official_resnet.pop('fc.weight', None)
            official_resnet.pop('fc.bias', None)
            model.backbone.load_state_dict(official_resnet)

    return model, optimizer


def save_model(path, model, optimizer=None):
    if optimizer is None:
        torch.save({'state_dict': model.state_dict()}, path)
    else:
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, path)

