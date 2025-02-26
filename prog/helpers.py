import logging
import os
import math
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from timm.utils import unwrap_model
from torch.nn import init
from torch.nn.init import trunc_normal_

from prog.progressive import make_divisible

_logger = logging.getLogger(__name__)


def load_state_dict(checkpoint_path, use_ema=False, ema_idx=None):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
            elif use_ema and ema_idx is not None and 'state_dict_ema_{}'.format(ema_idx) in checkpoint:
                state_dict_key = 'state_dict_ema_{}'.format(ema_idx)
            elif use_ema:
                _logger.info("No ema state dict found at '{}', fall back to online state dict".format(checkpoint_path))
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True, ema_idx=None):
    if os.path.splitext(checkpoint_path)[-1].lower() in ('.npz', '.npy'):
        # numpy checkpoint, try to load via model specific load_pretrained fn
        if hasattr(model, 'load_pretrained'):
            model.load_pretrained(checkpoint_path)
        else:
            raise NotImplementedError('Model cannot load numpy checkpoint')
        return
    state_dict = load_state_dict(checkpoint_path, use_ema, ema_idx=ema_idx)
    model.load_state_dict(state_dict, strict=strict)


def resume_checkpoint(model, checkpoint_path, optimizer=None, loss_scaler=None, log_info=True, use_ema=False, ema_idx=None):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
            elif use_ema and ema_idx is not None and 'state_dict_ema_{}'.format(ema_idx) in checkpoint:
                state_dict_key = 'state_dict_ema_{}'.format(ema_idx)
            elif use_ema:
                _logger.info("No ema state dict found at '{}', fall back to online state dict".format(checkpoint_path))
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            if log_info:
                _logger.info('Restoring model state from checkpoint {}...'.format(state_dict_key))
            model.load_state_dict(new_state_dict)

            if optimizer is not None and 'optimizer' in checkpoint:
                if log_info:
                    _logger.info('Restoring optimizer state from checkpoint...')
                optimizer.load_state_dict(checkpoint['optimizer'])

            if loss_scaler is not None and loss_scaler.state_dict_key in checkpoint:
                if log_info:
                    _logger.info('Restoring AMP loss scaler state from checkpoint...')
                loss_scaler.load_state_dict(checkpoint[loss_scaler.state_dict_key])

            if 'epoch' in checkpoint:
                resume_epoch = checkpoint['epoch']
                if 'version' in checkpoint and checkpoint['version'] > 1:
                    resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save

            if log_info:
                _logger.info("Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
        else:
            model.load_state_dict(checkpoint)
            if log_info:
                _logger.info("Loaded checkpoint '{}'".format(checkpoint_path))
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def get_resume_epoch(checkpoint_path):
    resume_epoch = None
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'epoch' in checkpoint:
            resume_epoch = checkpoint['epoch']
            if 'version' in checkpoint and checkpoint['version'] > 1:
                resume_epoch += 1  # start at the next epoch, old checkpoints incremented before save
        return resume_epoch
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_slice(model, checkpoint_model):
    print('Warning: loading in this mode is not implement correctly!! qkv is wrongly loaded!!')
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            if n in input_m_dict:
                input_m = input_m_dict[n]
                if isinstance(m, nn.Conv2d):
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    m.weight[:outc, :inc,
                    (mks - ks) // 2:(mks + ks) // 2,
                    (mks - ks) // 2:(mks + ks) // 2] = input_m.weight
                    if input_m.bias is not None:
                        m.bias[:outc] = input_m.bias
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    c = input_m.num_features
                    if m.affine:
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_channels
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        c = input_m.normalized_shape[0]
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    outc = input_m.out_features
                    inc = input_m.in_features
                    m.weight[:outc, :inc] = input_m.weight
                    if input_m.bias is not None:
                        m.bias[:outc] = input_m.bias
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    embed_dim = input_m.pos_embed.shape[-1]
                    m.pos_embed[..., :embed_dim] = input_m.pos_embed
                    if hasattr(m, 'cls_token'):
                        m.cls_token[..., :embed_dim] = input_m.cls_token
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            # elif len(list(m.parameters(recurse=False))) > 0:
                # print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model


def load_slice_clone_rand(model, checkpoint_model):
    print('Warning: loading in this mode is not implement correctly!! qkv is wrongly loaded!!')
    checkpoint_model = unwrap_model(checkpoint_model)
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            input_n = n
            if n.startswith('volo.network.0.') or n.startswith('volo.network.2.') or n.startswith('volo.network.3.') or n.startswith('volo.network.4.'):
                stage_idx = int(n.split('.')[2])
                if len(model.network[stage_idx]) > len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    input_layer_idx = new_idx(layer_idx, len(checkpoint_model.network[stage_idx]), len(model.network[stage_idx]))
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])

            if input_n in input_m_dict:
                input_m = input_m_dict[input_n]
                if isinstance(m, nn.Conv2d):
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    m.weight[:outc, :inc,
                    (mks - ks) // 2:(mks + ks) // 2,
                    (mks - ks) // 2:(mks + ks) // 2] = input_m.weight
                    if input_m.bias is not None:
                        m.bias[:outc] = input_m.bias
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    c = input_m.num_features
                    if m.affine:
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_channels
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        c = input_m.normalized_shape[0]
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    outc = input_m.out_features
                    inc = input_m.in_features
                    m.weight[:outc, :inc] = input_m.weight
                    if input_m.bias is not None:
                        m.bias[:outc] = input_m.bias
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    embed_dim = input_m.pos_embed.shape[-1]
                    m.pos_embed[..., :embed_dim] = input_m.pos_embed
                    if hasattr(m, 'cls_token'):
                        m.cls_token[..., :embed_dim] = input_m.cls_token
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            elif len(list(m.parameters(recurse=False))) > 0:
                print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model


def new_idx(idx, prev_l, new_l):
    if idx * prev_l // (new_l // prev_l * prev_l) < (prev_l - new_l % prev_l):
        return idx * prev_l // (new_l // prev_l * prev_l)
    else:
        return (idx + (prev_l - new_l % prev_l)) * prev_l // (new_l // prev_l * prev_l + prev_l)


def get_new_layer_idx(prev_l, new_l):
    return [i for i in range(new_l) if new_idx(i, prev_l, new_l) == new_idx(i - 1, prev_l, new_l)]


def load_slice_clone_zero(model, checkpoint_model, debug=False):
    checkpoint_model = unwrap_model(checkpoint_model)
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            input_n = n
            if n.startswith('volo.network.0.') or n.startswith('volo.network.2.') or n.startswith('volo.network.3.') or n.startswith('volo.network.4.'):
                stage_idx = int(n.split('.')[2])
                if len(model.network[stage_idx]) > len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    input_layer_idx = new_idx(layer_idx, len(checkpoint_model.network[stage_idx]), len(model.network[stage_idx]))
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])
            if debug:
                print(n, input_n)

            if input_n in input_m_dict:
                input_m = input_m_dict[input_n]
                if isinstance(m, nn.Conv2d):
                    init.zeros_(m.weight)
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    m.weight[:outc, :inc,
                    (mks - ks) // 2:(mks + ks) // 2,
                    (mks - ks) // 2:(mks + ks) // 2] = input_m.weight
                    if input_m.bias is not None:
                        init.zeros_(m.bias)
                        m.bias[:outc] = input_m.bias
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    c = input_m.num_features
                    if m.affine:
                        init.zeros_(m.weight)  # zero init weights when growing
                        init.zeros_(m.bias)
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        init.zeros_(m.weight)  # zero init weights when growing
                        init.zeros_(m.bias)
                        c = input_m.num_channels
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        init.zeros_(m.weight)  # zero init weights when growing
                        init.zeros_(m.bias)
                        c = input_m.normalized_shape[0]
                        m.weight[:c] = input_m.weight
                        m.bias[:c] = input_m.bias
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    init.zeros_(m.weight)
                    outc = input_m.out_features
                    inc = input_m.in_features
                    m.weight[:outc, :inc] = input_m.weight
                    if input_m.bias is not None:
                        init.zeros_(m.bias)
                        m.bias[:outc] = input_m.bias
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    init.zeros_(m.pos_embed)
                    embed_dim = input_m.pos_embed.shape[-1]
                    m.pos_embed[..., :embed_dim] = input_m.pos_embed
                    if hasattr(m, 'cls_token'):
                        init.zeros_(m.cls_token)
                        m.cls_token[..., :embed_dim] = input_m.cls_token
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            elif len(list(m.parameters(recurse=False))) > 0:
                print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model


def load_slice_clone(model, checkpoint_model, debug=False):
    checkpoint_model = unwrap_model(checkpoint_model)
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            input_n = n
            if n.startswith('volo.network.0.') or n.startswith('volo.network.2.') or n.startswith('volo.network.3.') or n.startswith('volo.network.4.'):
                stage_idx = int(n.split('.')[2])
                if len(model.network[stage_idx]) > len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    input_layer_idx = new_idx(layer_idx, len(checkpoint_model.network[stage_idx]), len(model.network[stage_idx]))
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])
            if debug:
                print(n, input_n)

            if input_n in input_m_dict:
                input_m = input_m_dict[input_n]
                if isinstance(m, nn.Conv2d):
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    assert ks == mks
                    minc = m.in_channels
                    moutc = m.out_channels
                    outc_repeat = moutc//outc + 1
                    inc_repeat = minc//inc + 1
                    weight_repeat = torch.cat([input_m.weight.clone() for _ in range(inc_repeat)], dim=1)
                    weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=0)
                    if n.startswith('volo.network.1.'):
                        scale = minc / inc
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    else:
                        m.weight[:] = weight_repeat[:moutc, :minc]
                    if input_m.bias is not None:
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(outc_repeat)], dim=0)
                        m.bias[:] = bias_repeat[:moutc]
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_features
                        mc = m.num_features
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_channels
                        mc = m.num_channels
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        c = input_m.normalized_shape[0]
                        mc = m.normalized_shape[0]
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    outc = input_m.out_features
                    inc = input_m.in_features
                    minc = m.in_features
                    moutc = m.out_features
                    outc_repeat = moutc // outc + 1
                    inc_repeat = minc // inc + 1
                    scale = minc / inc
                    if n.endswith('.qkv'):
                        assert outc % 3 == 0 and moutc % 3 == 0
                        weight_repeat = input_m.weight.clone().view(3, outc//3, inc)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(inc_repeat)], dim=2)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//3, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    elif n.endswith('.kv'):
                        assert outc % 2 == 0 and moutc % 2 == 0
                        weight_repeat = input_m.weight.clone().view(2, outc//2, inc)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(inc_repeat)], dim=2)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//2, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    else:
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(inc_repeat)], dim=1)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    if input_m.bias is not None:
                        if n.endswith('.qkv'):
                            bias_repeat = input_m.bias.clone().view(3, outc // 3)
                            bias_repeat = torch.cat([bias_repeat.clone() for _ in range(inc_repeat)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 3]
                            m.bias[:] = bias_repeat.view(moutc)
                        elif n.endswith('.kv'):
                            bias_repeat = input_m.bias.clone().view(2, outc // 2)
                            bias_repeat = torch.cat([bias_repeat.clone() for _ in range(inc_repeat)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 2]
                            m.bias[:] = bias_repeat.view(moutc)
                        else:
                            bias_repeat = torch.cat([input_m.bias.clone() for _ in range(outc_repeat)], dim=0)
                            m.bias[:] = bias_repeat[:moutc]
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    embed_dim = input_m.pos_embed.shape[-1]
                    m_embed_dim = m.pos_embed.shape[-1]
                    c_repeat = m_embed_dim // embed_dim + 1
                    pos_embed_repeat = torch.cat([input_m.pos_embed.clone() for _ in range(c_repeat)], dim=-1)
                    m.pos_embed[:] = pos_embed_repeat[..., :m_embed_dim]
                    if hasattr(m, 'cls_token'):
                        cls_token_repeat = torch.cat([input_m.cls_token.clone() for _ in range(c_repeat)], dim=-1)
                        m.cls_token[:] = cls_token_repeat[..., :m_embed_dim]
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            elif len(list(m.parameters(recurse=False))) > 0:
                print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model


def load_slice_clone_noise(model, checkpoint_model, debug=False):
    checkpoint_model = unwrap_model(checkpoint_model)
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            input_n = n
            if n.startswith('volo.network.0.') or n.startswith('volo.network.2.') or n.startswith('volo.network.3.') or n.startswith('volo.network.4.'):
                stage_idx = int(n.split('.')[2])
                if len(model.network[stage_idx]) > len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    input_layer_idx = new_idx(layer_idx, len(checkpoint_model.network[stage_idx]), len(model.network[stage_idx]))
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])
            if debug:
                print(n, input_n)

            if input_n in input_m_dict:
                input_m = input_m_dict[input_n]
                if isinstance(m, nn.Conv2d):
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    assert ks == mks
                    minc = m.in_channels
                    moutc = m.out_channels
                    outc_repeat = moutc//outc + 1
                    inc_repeat = minc//inc + 1
                    weight_repeat = torch.cat([input_m.weight.clone()] + [(input_m.weight.clone() + trunc_normal_(input_m.weight.clone(), std=.01)) for _ in range(inc_repeat)], dim=1)
                    weight_repeat = torch.cat([weight_repeat.clone()] + [(weight_repeat.clone() + trunc_normal_(weight_repeat.clone(), std=.01)) for _ in range(outc_repeat)], dim=0)
                    if n.startswith('volo.network.1.'):
                        scale = minc / inc
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    else:
                        m.weight[:] = weight_repeat[:moutc, :minc]
                    if input_m.bias is not None:
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(outc_repeat)], dim=0)
                        m.bias[:] = bias_repeat[:moutc]
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_features
                        mc = m.num_features
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_channels
                        mc = m.num_channels
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        c = input_m.normalized_shape[0]
                        mc = m.normalized_shape[0]
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    outc = input_m.out_features
                    inc = input_m.in_features
                    minc = m.in_features
                    moutc = m.out_features
                    outc_repeat = moutc // outc + 1
                    inc_repeat = minc // inc + 1
                    scale = minc / inc
                    if n.endswith('.qkv'):
                        assert outc % 3 == 0 and moutc % 3 == 0
                        weight_repeat = input_m.weight.clone().view(3, outc//3, inc)
                        weight_repeat = torch.cat([weight_repeat.clone()] + [(weight_repeat.clone() + trunc_normal_(weight_repeat.clone(), std=.01)) for _ in range(inc_repeat)], dim=2)
                        weight_repeat = torch.cat([weight_repeat.clone()] + [(weight_repeat.clone() + trunc_normal_(weight_repeat.clone(), std=.01)) for _ in range(outc_repeat)], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//3, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    elif n.endswith('.kv'):
                        assert outc % 2 == 0 and moutc % 2 == 0
                        weight_repeat = input_m.weight.clone().view(2, outc//2, inc)
                        weight_repeat = torch.cat([weight_repeat.clone()] + [(weight_repeat.clone() + trunc_normal_(weight_repeat.clone(), std=.01)) for _ in range(inc_repeat)], dim=2)
                        weight_repeat = torch.cat([weight_repeat.clone()] + [(weight_repeat.clone() + trunc_normal_(weight_repeat.clone(), std=.01)) for _ in range(outc_repeat)], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//2, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    else:
                        weight_repeat = torch.cat([input_m.weight.clone()] + [(input_m.weight.clone() + trunc_normal_(input_m.weight.clone(), std=.01)) for _ in range(inc_repeat)], dim=1)
                        weight_repeat = torch.cat([weight_repeat.clone()] + [(weight_repeat.clone() + trunc_normal_(weight_repeat.clone(), std=.01)) for _ in range(outc_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    if input_m.bias is not None:
                        if n.endswith('.qkv'):
                            bias_repeat = input_m.bias.clone().view(3, outc // 3)
                            bias_repeat = torch.cat([bias_repeat.clone() for _ in range(inc_repeat)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 3]
                            m.bias[:] = bias_repeat.view(moutc)
                        elif n.endswith('.kv'):
                            bias_repeat = input_m.bias.clone().view(2, outc // 2)
                            bias_repeat = torch.cat([bias_repeat.clone() for _ in range(inc_repeat)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 2]
                            m.bias[:] = bias_repeat.view(moutc)
                        else:
                            bias_repeat = torch.cat([input_m.bias.clone() for _ in range(outc_repeat)], dim=0)
                            m.bias[:] = bias_repeat[:moutc]
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    embed_dim = input_m.pos_embed.shape[-1]
                    m_embed_dim = m.pos_embed.shape[-1]
                    c_repeat = m_embed_dim // embed_dim + 1
                    pos_embed_repeat = torch.cat([input_m.pos_embed.clone() for _ in range(c_repeat)], dim=-1)
                    m.pos_embed[:] = pos_embed_repeat[..., :m_embed_dim]
                    if hasattr(m, 'cls_token'):
                        cls_token_repeat = torch.cat([input_m.cls_token.clone() for _ in range(c_repeat)], dim=-1)
                        m.cls_token[:] = cls_token_repeat[..., :m_embed_dim]
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            elif len(list(m.parameters(recurse=False))) > 0:
                print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model


def load_slice_clone_ema(model, checkpoint_model, ema_model_list, debug=False):
    checkpoint_model = unwrap_model(checkpoint_model)
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    assert len(ema_model_list) > 3
    input_ema_dict_l = [dict(unwrap_model(ema_model).named_modules(prefix='volo')) for ema_model in ema_model_list]
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            input_n = n
            if n.startswith('volo.network.0.') or n.startswith('volo.network.2.') or n.startswith('volo.network.3.') or n.startswith('volo.network.4.'):
                stage_idx = int(n.split('.')[2])
                if len(model.network[stage_idx]) > len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    input_layer_idx = new_idx(layer_idx, len(checkpoint_model.network[stage_idx]), len(model.network[stage_idx]))
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])
            if debug:
                print(n, input_n)

            if input_n in input_m_dict:
                input_m = input_m_dict[input_n]
                input_ema_l = [input_ema_dict[input_n] for input_ema_dict in input_ema_dict_l]
                if isinstance(m, nn.Conv2d):
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    assert ks == mks
                    minc = m.in_channels
                    moutc = m.out_channels
                    assert moutc <= 2 * outc and minc <= 2 * inc
                    weight_repeat1 = torch.cat([input_m.weight.clone()] + [input_ema_l[0].weight.clone()], dim=1)
                    weight_repeat2 = torch.cat([input_ema_l[1].weight.clone()] + [input_ema_l[2].weight.clone()], dim=1)
                    weight_repeat = torch.cat([weight_repeat1.clone()] + [weight_repeat2.clone()], dim=0)
                    if n.startswith('volo.network.1.'):
                        scale = minc / inc
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    else:
                        m.weight[:] = weight_repeat[:moutc, :minc]
                    if input_m.bias is not None:
                        bias_repeat = torch.cat([input_m.bias.clone()] + [input_ema_l[0].bias.clone()], dim=0)
                        m.bias[:] = bias_repeat[:moutc]
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_features
                        mc = m.num_features
                        assert mc <= 2 * c
                        weight_repeat = torch.cat([input_m.weight.clone()] + [input_ema_l[0].weight.clone()], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone()] + [input_ema_l[0].bias.clone()], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_channels
                        mc = m.num_channels
                        assert mc <= 2 * c
                        weight_repeat = torch.cat([input_m.weight.clone()] + [input_ema_l[0].weight.clone()], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone()] + [input_ema_l[0].bias.clone()], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        c = input_m.normalized_shape[0]
                        mc = m.normalized_shape[0]
                        assert mc <= 2 * c
                        weight_repeat = torch.cat([input_m.weight.clone()] + [input_ema_l[0].weight.clone()], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone()] + [input_ema_l[0].bias.clone()], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    outc = input_m.out_features
                    inc = input_m.in_features
                    minc = m.in_features
                    moutc = m.out_features
                    assert moutc <= 2 * outc and minc <= 2 * inc
                    scale = minc / inc
                    if n.endswith('.qkv'):
                        assert outc % 3 == 0 and moutc % 3 == 0
                        weight_repeat1 = torch.cat([input_m.weight.clone().view(3, outc//3, inc),
                                                    input_ema_l[0].weight.clone().view(3, outc//3, inc)], dim=2)
                        weight_repeat2 = torch.cat([input_ema_l[1].weight.clone().view(3, outc//3, inc),
                                                    input_ema_l[2].weight.clone().view(3, outc//3, inc)], dim=2)
                        weight_repeat = torch.cat([weight_repeat1.clone(), weight_repeat2.clone()], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//3, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    elif n.endswith('.kv'):
                        assert outc % 2 == 0 and moutc % 2 == 0
                        weight_repeat1 = torch.cat([input_m.weight.clone().view(2, outc // 2, inc),
                                                    input_ema_l[0].weight.clone().view(2, outc // 2, inc)], dim=2)
                        weight_repeat2 = torch.cat([input_ema_l[1].weight.clone().view(2, outc // 2, inc),
                                                    input_ema_l[2].weight.clone().view(2, outc // 2, inc)], dim=2)
                        weight_repeat = torch.cat([weight_repeat1.clone(), weight_repeat2.clone()], dim=1)
                        weight_repeat = weight_repeat[:, :moutc // 2, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    else:
                        weight_repeat1 = torch.cat([input_m.weight.clone(), input_ema_l[0].weight.clone()], dim=1)
                        weight_repeat2 = torch.cat([input_ema_l[1].weight.clone(), input_ema_l[2].weight.clone()], dim=1)
                        weight_repeat = torch.cat([weight_repeat1.clone(), weight_repeat2.clone()], dim=0)
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    if input_m.bias is not None:
                        if n.endswith('.qkv'):
                            bias_repeat = torch.cat([input_m.bias.clone().view(3, outc // 3),
                                                     input_ema_l[0].bias.clone().view(3, outc // 3)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 3]
                            m.bias[:] = bias_repeat.view(moutc)
                        elif n.endswith('.kv'):
                            bias_repeat = torch.cat([input_m.bias.clone().view(2, outc // 2),
                                                     input_ema_l[0].bias.clone().view(2, outc // 2)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 2]
                            m.bias[:] = bias_repeat.view(moutc)
                        else:
                            bias_repeat = torch.cat([input_m.bias.clone(), input_ema_l[0].bias.clone()], dim=0)
                            m.bias[:] = bias_repeat[:moutc]
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    embed_dim = input_m.pos_embed.shape[-1]
                    m_embed_dim = m.pos_embed.shape[-1]
                    assert m_embed_dim <= 2 * embed_dim
                    pos_embed_repeat = torch.cat([input_m.pos_embed.clone(), input_ema_l[0].pos_embed.clone()], dim=-1)
                    m.pos_embed[:] = pos_embed_repeat[..., :m_embed_dim]
                    if hasattr(m, 'cls_token'):
                        cls_token_repeat = torch.cat([input_m.cls_token.clone(), input_ema_l[0].cls_token.clone()], dim=-1)
                        m.cls_token[:] = cls_token_repeat[..., :m_embed_dim]
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            elif len(list(m.parameters(recurse=False))) > 0:
                print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model


def load_super(model, checkpoint_model, base_layer, model_name, debug=False):
    if model_name=='deit':  # deit_h12_l12  TODO: deit
        base_layer = [base_layer]
    elif model_name=='volo':  # volo_h12_l18
        if base_layer > 2:
            l0 = make_divisible(base_layer * 0.23, 2)
            base_layer = [l0, 0, base_layer - l0, 0, 0]  # stage 1 is down-sample
        else:
            base_layer = [1, 0, 1, 0, 0]

    checkpoint_model = unwrap_model(checkpoint_model)
    input_m_dict = dict(unwrap_model(checkpoint_model).named_modules(prefix='volo'))
    with torch.no_grad():
        for n, m in model.named_modules(prefix='volo'):
            # print('===========', n, ':', type(m))
            input_n = n
            if n.startswith('volo.network.0.') or n.startswith('volo.network.2.') or n.startswith('volo.network.3.') or n.startswith('volo.network.4.'):
                stage_idx = int(n.split('.')[2])
                if len(model.network[stage_idx]) > len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    input_layer_idx = new_idx(layer_idx, len(checkpoint_model.network[stage_idx]), len(model.network[stage_idx]))
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])
                elif len(model.network[stage_idx]) < len(checkpoint_model.network[stage_idx]):
                    layer_idx = int(n.split('.')[3])
                    new_layer_idxs = get_new_layer_idx(prev_l=base_layer[stage_idx], new_l=len(checkpoint_model.network[stage_idx]))
                    if len(model.network[stage_idx]) - base_layer[stage_idx] > 0:
                        skip_layer_idxs = new_layer_idxs[:-(len(model.network[stage_idx]) - base_layer[stage_idx])]
                    elif len(model.network[stage_idx]) - base_layer[stage_idx] == 0:
                        skip_layer_idxs = new_layer_idxs
                    no_skip_layer_idxs = [idx for idx in range(len(checkpoint_model.network[stage_idx])) if idx not in skip_layer_idxs]
                    assert len(model.network[stage_idx]) == len(no_skip_layer_idxs), \
                        f'{len(model.network[stage_idx])} and {len(no_skip_layer_idxs)} must be equal'
                    input_layer_idx = no_skip_layer_idxs[layer_idx]
                    input_n = '.'.join(n.split('.')[:3] + [str(input_layer_idx)] + n.split('.')[4:])
            if debug:
                print(n, input_n)

            if input_n in input_m_dict:
                input_m = input_m_dict[input_n]
                if isinstance(m, nn.Conv2d):
                    # print('load:', n, type(m))
                    inc = input_m.in_channels
                    outc = input_m.out_channels
                    ks = input_m.kernel_size[0]
                    mks = m.kernel_size[0]
                    assert ks == mks
                    minc = m.in_channels
                    moutc = m.out_channels
                    outc_repeat = moutc//outc + 1
                    inc_repeat = minc//inc + 1
                    weight_repeat = torch.cat([input_m.weight.clone() for _ in range(inc_repeat)], dim=1)
                    weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=0)
                    if n.startswith('volo.network.1.'):
                        scale = minc / inc
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    else:
                        m.weight[:] = weight_repeat[:moutc, :minc]
                    if input_m.bias is not None:
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(outc_repeat)], dim=0)
                        m.bias[:] = bias_repeat[:moutc]
                elif isinstance(m, nn.BatchNorm2d):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_features
                        mc = m.num_features
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                    # if m.track_running_stats:  # TODO: check if this influence result
                    #     m.running_mean[:c] = input_m.running_mean
                    #     m.running_var[:c] = input_m.running_var
                    #     m.num_batches_tracked = input_m.num_batches_tracked
                elif isinstance(m, nn.GroupNorm):
                    # print('load:', n, type(m))
                    if m.affine:
                        c = input_m.num_channels
                        mc = m.num_channels
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                        m.num_groups = m.num_channels // (c // input_m.num_groups)
                elif isinstance(m, nn.LayerNorm):
                    # print('load:', n, type(m))
                    if m.elementwise_affine:
                        c = input_m.normalized_shape[0]
                        mc = m.normalized_shape[0]
                        c_repeat = mc // c + 1
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(c_repeat)], dim=0)
                        bias_repeat = torch.cat([input_m.bias.clone() for _ in range(c_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:mc]
                        m.bias[:] = bias_repeat[:mc]
                elif isinstance(m, nn.Linear):
                    # print('load:', n, type(m))
                    outc = input_m.out_features
                    inc = input_m.in_features
                    minc = m.in_features
                    moutc = m.out_features
                    outc_repeat = moutc // outc + 1
                    inc_repeat = minc // inc + 1
                    scale = minc / inc
                    if n.endswith('.qkv'):
                        assert outc % 3 == 0 and moutc % 3 == 0
                        weight_repeat = input_m.weight.clone().view(3, outc//3, inc)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(inc_repeat)], dim=2)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//3, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    elif n.endswith('.kv'):
                        assert outc % 2 == 0 and moutc % 2 == 0
                        weight_repeat = input_m.weight.clone().view(2, outc//2, inc)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(inc_repeat)], dim=2)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=1)
                        weight_repeat = weight_repeat[:, :moutc//2, :minc] / scale
                        m.weight[:] = weight_repeat.view(moutc, minc)
                    else:
                        weight_repeat = torch.cat([input_m.weight.clone() for _ in range(inc_repeat)], dim=1)
                        weight_repeat = torch.cat([weight_repeat.clone() for _ in range(outc_repeat)], dim=0)
                        m.weight[:] = weight_repeat[:moutc, :minc] / scale
                    if input_m.bias is not None:
                        if n.endswith('.qkv'):
                            bias_repeat = input_m.bias.clone().view(3, outc // 3)
                            bias_repeat = torch.cat([bias_repeat.clone() for _ in range(inc_repeat)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 3]
                            m.bias[:] = bias_repeat.view(moutc)
                        elif n.endswith('.kv'):
                            bias_repeat = input_m.bias.clone().view(2, outc // 2)
                            bias_repeat = torch.cat([bias_repeat.clone() for _ in range(inc_repeat)], dim=1)
                            bias_repeat = bias_repeat[:, :moutc // 2]
                            m.bias[:] = bias_repeat.view(moutc)
                        else:
                            bias_repeat = torch.cat([input_m.bias.clone() for _ in range(outc_repeat)], dim=0)
                            m.bias[:] = bias_repeat[:moutc]
                elif hasattr(m, 'pos_embed'):
                    # print('load:', n, type(m))
                    embed_dim = input_m.pos_embed.shape[-1]
                    m_embed_dim = m.pos_embed.shape[-1]
                    c_repeat = m_embed_dim // embed_dim + 1
                    pos_embed_repeat = torch.cat([input_m.pos_embed.clone() for _ in range(c_repeat)], dim=-1)
                    m.pos_embed[:] = pos_embed_repeat[..., :m_embed_dim]
                    if hasattr(m, 'cls_token'):
                        cls_token_repeat = torch.cat([input_m.cls_token.clone() for _ in range(c_repeat)], dim=-1)
                        m.cls_token[:] = cls_token_repeat[..., :m_embed_dim]
                elif len(list(input_m.parameters(recurse=False))) > 0:
                    raise NotImplementedError('method not implemented for {n}:{m}')
            elif len(list(m.parameters(recurse=False))) > 0:
                print('no module \'{}\' {} in slice.'.format(n, type(m)))
    return model, checkpoint_model