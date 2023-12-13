

from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
import sampling
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_fast)
from models import ncsnpp
import time
from utils import fft2, ifft2, get_mask, get_data_scaler, get_data_inverse_scaler, restore_checkpoint
import torch
import torch.nn as nn
import numpy as np
from models.ema import ExponentialMovingAverage
import matplotlib.pyplot as plt
import importlib
import argparse

import run


def sample_exp():

    config = run.get_configs()
    # config.device = torch.device('cpu')
    print('using device:',config.device)
    img_size = config.data.image_size
    channels = config.data.num_channels
    batch_size = 1
    shape = (batch_size, channels, img_size, img_size)
    predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
    corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
    snr = 0.16 #@param {"type": "number"}
    n_steps =  1#@param {"type": "integer"}
    probability_flow = False #@param {"type": "boolean"}


    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=2000)


    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # ckpt_filename = f"./weights/checkpoint_95.pth" # knee model
    ckpt_filename = f"./training_log/checkpoints/checkpoint_33.pth" # prostate model

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    # state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=False)
    ema.copy_to(score_model.parameters())

    sampling_eps = 1e-5
    sampling_fn = sampling.get_pc_sampler(sde, shape, predictor, corrector,
                                        inverse_scaler, snr, n_steps=n_steps,
                                        probability_flow=probability_flow,
                                        continuous=config.training.continuous,
                                        eps=sampling_eps, device=config.device)

    print('begin sampling...')
    x, n = sampling_fn(score_model)
    # show_samples(x)
    img = x.squeeze().cpu().detach().numpy()
    plt.imsave('test_sample.png', img, cmap='gray')
    
    return

if __name__=="__main__":
    sample_exp()
