import os
import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter


from lib.core.config import parse_args
from lib.core.losses import compile_criterion
from lib.utils.utils import prepare_output_dir, create_logger

from lib import get_model, get_dataloaders
from lib.trainer import Trainer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    # create logger
    logger = create_logger(cfg.LOGDIR, phase='train')
    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')
    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # Dataloaders
    data_loaders = get_dataloaders(cfg)

    # Compile Loss 
    criterion = compile_criterion(cfg)

    # Networks and optimizers
    model = get_model(cfg)

    model = model.to(cfg.DEVICE)
    model.frozen_modules = []
    model.freeze_modules()

    if cfg.TRAIN.OPT == 'AdamW':
        optimizer = torch.optim.AdamW(params=[p for p in model.parameters() if p.requires_grad], 
                                     lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WD)
    else:
        optimizer = torch.optim.Adam(params=[p for p in model.parameters() if p.requires_grad], 
                                     lr=cfg.TRAIN.LR, weight_decay=0.0)
        
        print('Using Adam with 0 weight decay.')


    # ========= Start Training ========= #
    Trainer(
        cfg=cfg,
        data_loaders=data_loaders,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        writer=writer,
        lr_scheduler=None,
    ).train()



if __name__ == '__main__':
    cfg = parse_args()
    cfg = prepare_output_dir(cfg)

    main(cfg)
