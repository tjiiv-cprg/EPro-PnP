"""
Copyright (C) 2010-2022 Alibaba Group Holding Limited.
This file is modified from
https://github.com/tjiiv-cprg/MonoRUn
"""

import os
import torch
from torch._six import inf

from mmcv.runner import HOOKS, Hook


def clip_grad_norm_(parameters, max_norm, norm_type):
    assert isinstance(parameters, list)
    assert len(parameters) > 0
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device)
                 for p in parameters]),
            norm_type)
    if total_norm.isnan() | total_norm.isinf():
        clip_coef = 0
        for p in parameters:
            p.grad.detach().fill_(0)
    else:
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm, clip_coef


def save_stats(default_names, default_params, special_names, special_params,
               grad_norms, work_dir, rank, iterate):
    grad_dir = os.path.join(work_dir, 'grad')
    try:
        os.mkdir(grad_dir)
    except FileExistsError:
        pass
    norms_str = '_'.join(['{:.2e}'.format(norm) for norm in grad_norms.values()])
    outfile_path = os.path.join(
        grad_dir, 'iter_{:06d}_{:d}_{}.txt'.format(iterate, rank, norms_str))
    outfile = open(outfile_path, 'w')
    for key, value in grad_norms.items():
        outfile.write('{} = {:.6f}\n'.format(key, value))
    outfile.write('\n{:>12} {:>12} {:>12}    {}\n'.format(
        'clipped_grad', 'var', 'mean', 'default'))
    for name, param in zip(default_names, default_params):
        clipped_rms = param.grad.detach().square().mean().sqrt()
        var, mean = torch.var_mean(param.detach())
        outfile.write('{:>12.6f} {:>12.6f} {:>12.6f}    {}\n'.format(
            clipped_rms, var.sqrt(), mean, name))
    for key in special_names.keys():
        outfile.write('\n{:>12} {:>12} {:>12}    {}\n'.format(
            'clipped_grad', 'var', 'mean', key))
        for name, param in zip(special_names[key], special_params[key]):
            clipped_rms = param.grad.detach().square().mean().sqrt()
            var, mean = torch.var_mean(param.detach())
            outfile.write('{:>12.6f} {:>12.6f} {:>12.6f}    {}\n'.format(
                clipped_rms, var.sqrt(), mean, name))
    outfile.close()


@HOOKS.register_module()
class OptimizerHookMod(Hook):

    def __init__(self,
                 grad_clip=None,
                 save_stats_interval=-1,
                 save_stats_clipped=False):
        self.grad_clip = grad_clip
        self.grad_clip_paramwise_cfg = self.grad_clip.pop('paramwise_cfg', dict())
        self.save_stats_interval = save_stats_interval
        self.save_stats_clipped = save_stats_clipped

    def clip_grads(self, named_params, work_dir, iterate, rank):
        default_names, default_params, special_names, special_params = \
            self.params_filter(named_params)
        grad_norms = dict()
        clipped = False
        if len(default_params) > 0:
            norm, clip_coef = clip_grad_norm_(default_params, **self.grad_clip)
            grad_norms.update(default_grad_norm=float(norm))
            if clip_coef < 1:
                clipped = True
        for key, grad_clip_cfg in self.grad_clip_paramwise_cfg.items():
            params = special_params[key]
            if len(params) > 0:
                norm, clip_coef = clip_grad_norm_(params, **grad_clip_cfg)
                grad_norms.update({key + '_grad_norm': float(norm)})
                if clip_coef < 1:
                    clipped = True
        if self.will_save_stats(iterate, clipped):
            save_stats(default_names, default_params, special_names, special_params,
                       grad_norms, work_dir, rank, iterate)
        return grad_norms

    def will_save_stats(self, iterate, clipped):
        a = self.save_stats_interval > 0 and (iterate % self.save_stats_interval) == 0
        b = self.save_stats_clipped and clipped
        return a or b

    def params_filter(self, named_params):
        default_names = []
        default_params = []
        special_names = {key: [] for key in self.grad_clip_paramwise_cfg.keys()}
        special_params = {key: [] for key in self.grad_clip_paramwise_cfg.keys()}
        for name, param in named_params:
            if (param.grad is None) or (not param.requires_grad):
                continue
            default = True
            for key in self.grad_clip_paramwise_cfg.keys():
                if key in name:
                    default = False
                    special_names[key].append(name)
                    special_params[key].append(param)
                    break
            if default:
                default_names.append(name)
                default_params.append(param)
        return default_names, default_params, special_names, special_params

    def after_train_iter(self, runner):
        runner.optimizer.zero_grad()
        runner.outputs['loss'].backward()
        if self.grad_clip is not None:
            grad_norms = self.clip_grads(runner.model.named_parameters(),
                                         runner.work_dir, runner.iter, runner.rank)
            if grad_norms is not None:
                # Add grad norm to the logger
                runner.log_buffer.update(grad_norms,
                                         runner.outputs['num_samples'])
        ret = runner.optimizer.step()
        if isinstance(ret, tuple):
            step_meanabs = ret[1]
            runner.log_buffer.update(dict(step_meanabs=float(step_meanabs)),
                                     runner.outputs['num_samples'])
