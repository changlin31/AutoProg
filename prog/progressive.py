import numpy as np


def progressive_schedule(args, r_max=224, h_max=12, l_max=18):
    num_stages = args.num_stages
    r_scale = args.r_scale
    h_scale = args.h_scale
    l_scale = args.l_scale
    aa_scale = args.aa_scale
    dp_scale = args.dp_scale
    re_scale = args.re_scale
    resize_scale = args.resize_scale
    aa_max = args.aa
    dp_max = args.drop_path
    re_max = args.reprob
    resize_max = args.scale

    epochs = args.epochs
    e = [int(i) for i in np.linspace(0, epochs, num_stages + 1) // 1][:-1]
    r = [make_divisible(i, 32) for i in np.linspace(r_scale, 1., num_stages) * r_max]
    h = [make_divisible(i, 2) for i in np.linspace(h_scale, 1., num_stages) * h_max]   # h must be divisible by 2
    l = [make_divisible(i, 1) for i in np.linspace(l_scale, 1., num_stages) * l_max]
    assert isinstance(aa_max, str) and aa_max.startswith('rand')
    m_aa_max = float(aa_max.split('-')[1].lstrip('m'))
    m_aa = [round(max(0., i)) for i in np.linspace(aa_scale, 1., num_stages) * m_aa_max]
    aa = ['rand-m{}-mstd0.5-inc1'.format(m) if m > 0 else '' for m in m_aa]
    dp = [max(0., i) for i in np.linspace(dp_scale, 1., num_stages) * dp_max]
    re = [max(0., i) for i in np.linspace(re_scale, 1., num_stages) * re_max]
    resize = [[max(0., i[0]), max(0., i[1])] for i in zip(np.linspace(resize_scale[0], 1., num_stages) * resize_max[0],
                                                          np.linspace(resize_scale[1], 1., num_stages) * resize_max[1])]
    return e, r, h, l, aa, dp, re, resize


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v
