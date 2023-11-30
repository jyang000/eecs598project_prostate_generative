from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
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


def main():
    #########################################
    # 1. Configuration

    N = None
    m = 1
    fname = ''
    filename = ''
    mask_filename = ''

    print('initializing...')
    config = run.get_configs()
    img_size = config.data.image_size
    batch_size = 1
    print('config.device:',config.device)


    # Read data
    if True:
        # Knee data sample
        img = torch.from_numpy(np.load(filename))
    else:
        # Prostate data sample
        filename = '/scratch/Projects/fastmri_prostate/0001.npy'
        img = torch.from_numpy(np.load(filename))
        img = img[4,:,:]
    img = img.view(1, 1, 320, 320)
    img = img.to(config.device)

    # 'retrospective':
    # generate mask
    mask = get_mask(img, img_size, batch_size,
                    type=args.mask_type,
                    acc_factor=args.acc_factor,
                    center_fraction=args.center_fraction)
    
    ckpt_filename = f"./weights/checkpoint_95.pth" # knee model
    # ckpt_filename = f"./training_log/checkpoints/checkpoint_3.pth" # prostate model
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    config.training.batch_size = batch_size
    predictor = ReverseDiffusionPredictor
    corrector = LangevinCorrector
    probability_flow = False
    snr = 0.16

    # sigmas = mutils.get_sigmas(config)
    scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # create model and load checkpoint
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),
                                   decay=config.model.ema_rate)
    state = dict(step=0, model=score_model, ema=ema)
    # state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True)
    state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=False)
    ema.copy_to(score_model.parameters())

    # Specify save directory for saving generated samples
    save_root = Path(f'./results/real')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'recon_progress', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)


    ###############################################
    # 2. Inference
    ###############################################

    pc_fouriercs = get_pc_fouriercs_fast(sde,
                                         predictor, corrector,
                                         inverse_scaler,
                                         snr=snr,
                                         n_steps=m,
                                         probability_flow=probability_flow,
                                         continuous=config.training.continuous,
                                         denoise=True,
                                         save_progress=True,
                                         save_root=save_root / 'recon_progress')
    # fft
    kspace = fft2(img)

    # undersampling
    under_kspace = kspace * mask
    under_img = torch.real(ifft2(under_kspace))

    # test:
    # print(mask.device,under_kspace.device) # cuda:0

    print(f'Beginning inference')
    tic = time.time()
    x = pc_fouriercs(score_model, scaler(under_img), mask, Fy=under_kspace)
    toc = time.time() - tic
    print(f'Time took for recon: {toc} secs.')

    ###############################################
    # 3. Saving recon
    ###############################################

    if args.task == 'retrospective':
        # save input and label only if this is retrospective recon.
        input = under_img.squeeze().cpu().detach().numpy()
        label = img.squeeze().cpu().detach().numpy()
        mask_sv = mask.squeeze().cpu().detach().numpy()

        np.save(str(save_root / 'input' / fname) + '.npy', input)
        np.save(str(save_root / 'input' / (fname + '_mask')) + '.npy', mask_sv)
        np.save(str(save_root / 'label' / fname) + '.npy', label)
        plt.imsave(str(save_root / 'input' / fname) + '.png', input, cmap='gray')
        plt.imsave(str(save_root / 'label' / fname) + '.png', label, cmap='gray')

    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', recon, cmap='gray')
