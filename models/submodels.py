from functools import partial
import torch.nn as nn
from prog.progressive import make_divisible
from .volo import default_cfgs, VOLO
from timm.models.registry import register_model
from timm.models.vision_transformer import VisionTransformer, _cfg


@register_model
def model_variant(variant='', pretrained=False, **kwargs):
    """
    a general model with variants of any sizes
    """
    assert variant == 'volo_h12_l18', 'only support volo_h12_l18 for now'

    h = int(variant.split('_')[1].lstrip('h'))
    l = int(variant.split('_')[2].lstrip('l'))
    assert h % 2 == 0, 'h must be divisible by 2'

    if l > 2:
        l0 = make_divisible(l * 0.23, 2)
        layers = [l0, l - l0, 0, 0]
    else:
        print('Warning: layer too small, set to 2')
        layers = [1, 1, 0, 0]

    embed_dims = [h*16, h*32, h*32, h*32]
    num_heads = [h//2, h, h, h]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, False, False, False]
    outlook_attention = [True, False, False, False]
    model = VOLO(layers,
                 embed_dims=embed_dims,
                 num_heads=num_heads,
                 mlp_ratios=mlp_ratios,
                 downsamples=downsamples,
                 outlook_attention=outlook_attention,
                 post_layers=['ca', 'ca'],
                 **kwargs)
    model.default_cfg = default_cfgs['volo']
    return model