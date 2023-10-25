import os
import torch
from lib.models.refit import REFIT
from lib.models.refit_mv import REFIT_MV


def get_model(cfg):

    # get parameters
    model_params = cfg.MODEL
    model_params = {k.lower():v for k,v in model_params.items()}

    params = model_params
    params['device'] = cfg.DEVICE
    params['cfg'] = cfg


    if model_params['version'] == 'refit':
        model = REFIT(**params)
        print('model refit')

    elif model_params['version'] == 'mv':
        model = REFIT_MV(**params)
        print('model multiview refit')

    else:
        print('Model version not available!!')


    return model


def get_trained_model(cfg):
    model = get_model(cfg)
    cfg_id = cfg.EXP_NAME.split('_')[-1]

    checkpoint_file = 'results/refit_{}/checkpoint_best.pth.tar'.format(cfg_id)
    if os.path.exists(checkpoint_file):
        checkpoint = torch.load(checkpoint_file, map_location=cfg.DEVICE)
        _ = model.load_state_dict(checkpoint['model'], strict=False)
        print('Loaded model from', checkpoint_file)
    else:
        print('No checkpoint from', checkpoint_file)
        print('Loaded randomly initialized model.')


    return model

    
