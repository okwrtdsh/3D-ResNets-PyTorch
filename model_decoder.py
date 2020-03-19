import torch
from torch import nn

from models.decoder import STSRResNetExp, SVSTSRResNetExp, TSRResNetExp, SVTSRResNetExp

def generate_model(opt):
    assert opt.model in [
        'stsrresnetexp',
        'svstsrresnetexp',
        'tsrresnetexp',
        'svtsrresnetexp',
    ]

    if opt.model == 'stsrresnetexp':
        model = STSRResNetExp(
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'svstsrresnetexp':
        model = SVSTSRResNetExp(
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'tsrresnetexp':
        model = TSRResNetExp(
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)
    elif opt.model == 'svtsrresnetexp':
        model = SVTSRResNetExp(
            sample_size=opt.sample_size,
            sample_duration=opt.sample_duration)

    if not opt.no_cuda:
        model = model.cuda()
        model = nn.DataParallel(model, device_ids=None)
    return model, model.parameters()

