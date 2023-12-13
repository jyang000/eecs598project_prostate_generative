from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector,
                      get_pc_fouriercs_fast,
                      get_pc_fouriercs_fast_3d)
from models import ncsnpp
import time
from utils import (fft2, ifft2, get_mask, get_data_scaler, 
                   get_data_inverse_scaler, restore_checkpoint, 
                   fft3, ifft3, get_mask3d)

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
    args = create_argparser().parse_args()
    N = args.N
    m = 1
    fname = ''
    filename = ''
    # mask_filename = ''

    print('initializing...')
    config = run.get_configs()

    # config.data.image_size = 320 # for knee data

    img_size = config.data.image_size
    # batch_size = 1
    print('config.device:',config.device)
    print('image size:',img_size)

    def save_img3d(path,img):
        ns,ny,nx = img.shape
        nrow = 5
        ncol = int(ns//5)
        img_big = np.zeros((ny*nrow,nx*ncol))
        nim = 0
        for yid in range(nrow):
            for xid in range(ncol):
                img_big[yid*ny:(yid+1)*ny,xid*nx:(xid+1)*nx] = img[nim]
                nim = nim + 1
                if nim == ns:
                    break
            if nim == ns:
                break
        plt.imsave(path,img_big,cmap='gray')
        return


    # Read data
    if False:
        # Knee data sample
        filename = '/scratch/Projects/fastmri_prostate/knee3d.npy'
        img = torch.from_numpy(np.load(filename))*8e2
        img = img[6:36]
        print(img.shape)
    else:
        # Prostate data sample
        filename = '/scratch/Projects/fastmri_prostate/0001.npy'
        img = torch.from_numpy(np.load(filename))*3.3e3
        img = img[:,10:(10+256),32:(32+256)]
        # img_slice = img[4,:,:]
    # img_slice = img_slice.view(1, 1, 320, 320)
    # img_slice = img_slice.to(config.device)
    # print(img_slice.shape)
    # return

    nslices = config.data.nslices
    # then select small number of slices
    img = img[:nslices,:,:]
    # img_vslice = img[0,:,:]
    # img_vslice = img_vslice.view(1,1,nslices,320)
    # img_slice = img_slice.to(config.device)
    print('3D image shape:',img.shape,torch.mean(img),torch.max(img))

    # 'retrospective':
    # generate mask
    mask = get_mask3d(img,nx=img_size,ny=img_size,nz=nslices,acc_factor=16)
    # ---------------------------------
    # print('mask:',mask.shape)
    # # Save the images
    # mask2d_img = mask[:,:,0].squeeze().cpu().numpy()
    # plt.imsave('test/img1.png',mask2d_img,cmap='gray')
    # mask2d_img = mask[15,:,:].squeeze().cpu().numpy()
    # plt.imsave('test/img.png',mask2d_img,cmap='gray')
    # mask2d_img = mask[7,:,:].squeeze().cpu().numpy()
    # plt.imsave('test/img2.png',mask2d_img,cmap='gray')

    # illustrate of 3D undersamplin'
    # ---------------------------------
    # ax = plt.figure(figsize=(8,6)).add_subplot(projection='3d')
    # _,ny,nx = mask.shape
    # ptx = []
    # pty = []
    # ptz = []
    # idxs = np.nonzero(mask>0)
    # print(idxs)
    # # for xidx in range(nx):
    # #     for yidx in range(ny):
    # #         for slice in range(nslices):
    # #             if mask[slice,yidx,xidx] > 0:
    # #                 ptx.append(xidx)
    # #                 pty.append(yidx)
    # #                 ptz.append(slice)
    # ptx = idxs[:,1]
    # pty = idxs[:,2]
    # ptz = idxs[:,0]
    # ax.scatter(ptx,pty,ptz,color='black',s=0.05, alpha=0.5)
    # selectidx = np.nonzero(pty<4)
    # print('red:',len(selectidx))
    # ptx = ptx[selectidx]
    # pty = pty[selectidx]
    # ptz = ptz[selectidx]
    # ax.scatter(ptx,pty,ptz,color='red',s=1,)
    # # plt.axis("off")
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xlabel('ky')
    # ax.set_ylabel('kx')
    # ax.set_zlabel('kz')
    # # ax.set_aspect('equal', adjustable='box')
    # ax.view_init(elev=20., azim=-75, roll=0)
    # plt.savefig('test/mask3d.png')
    # return


    # Reshape the data and mask
    img = img.view(nslices,img_size,img_size).to(config.device)
    mask = mask.view(nslices,img_size,img_size).to(config.device)
    img = img.to(torch.float32)
    img = img.to(config.device)
    mask = mask.to(torch.float32)
    mask = mask.to(config.device)
    print(img.dtype,mask.dtype)

    
    # ckpt_filename = f"./weights/checkpoint_95.pth" # knee model
    ckpt_filename = f"./training_log_prostate_epoch45/checkpoints/checkpoint_45.pth" # prostate model
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=N)

    # config.training.batch_size = batch_size
    config.training.batch_size = nslices
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

    pc_fouriercs = get_pc_fouriercs_fast_3d(
        sde,
        predictor, corrector,
        inverse_scaler,
        snr=snr,
        n_steps=m,
        probability_flow=probability_flow,
        continuous=config.training.continuous,
        denoise=True,
        save_progress=True,
        save_root=save_root / 'recon_progress',
        nslices=nslices,img_size=img_size)
    # fft
    # kspace = fft2(img)
    kspace = fft3(img)

    # undersampling
    under_kspace = kspace * mask
    under_img = torch.real(ifft3(under_kspace))

    save_img3d(str(save_root / 'input' / fname) + '_kspacesampling.png',(under_kspace.abs().cpu().numpy()>0))
    save_img3d(str(save_root / 'input' / fname) + '_kspace.png',torch.log(under_kspace.abs()+1e-4).cpu().numpy())

    # test of functions

    # test:
    # print(mask.device,under_kspace.device) # cuda:0
    # return

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

        np.save(str(save_root / 'input' / fname) + '_3d.npy', input)
        np.save(str(save_root / 'input' / (fname + '_mask3d')) + '.npy', mask_sv)
        np.save(str(save_root / 'label' / fname) + '_3d.npy', label)
        plt.imsave(str(save_root / 'input' / fname) + '.png', input[15], cmap='gray')
        plt.imsave(str(save_root / 'label' / fname) + '.png', label[15], cmap='gray')
        save_img3d(str(save_root / 'input' / fname) + '.png', input)
        save_img3d(str(save_root / 'label' / fname) + '.png', label)

    recon = x.squeeze().cpu().detach().numpy()
    np.save(str(save_root / 'recon' / fname) + '_3d.npy', recon)
    plt.imsave(str(save_root / 'recon' / fname) + '.png', recon[15], cmap='gray')
    save_img3d(str(save_root / 'recon' / fname) + '.png', recon)


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['retrospective', 'prospective'], default='retrospective',
                        type=str, help='If retrospective, under-samples the fully-sampled data with generated mask.'
                                       'If prospective, runs score-POCS with the given mask')
    parser.add_argument('--data', type=str, default='001', help='which data to use for reconstruction', required=False)
    parser.add_argument('--mask_type', type=str, help='which mask to use for retrospective undersampling.'
                                                      '(NOTE) only used for retrospective model!', default='gaussian1d',
                        choices=['gaussian1d', 'uniform1d', 'gaussian2d'])
    parser.add_argument('--acc_factor', type=int, help='Acceleration factor for Fourier undersampling.'
                                                       '(NOTE) only used for retrospective model!', default=4)
    parser.add_argument('--center_fraction', type=float, help='Fraction of ACS region to keep.'
                                                       '(NOTE) only used for retrospective model!', default=0.08)
    parser.add_argument('--save_dir', default='./results')
    parser.add_argument('--N', type=int, help='Number of iterations for score-POCS sampling', default=2000)
    parser.add_argument('--m', type=int, help='Number of corrector step per single predictor step.'
                                              'It is advised not to change this default value.', default=1)
    return parser

if __name__=="__main__":
    main()