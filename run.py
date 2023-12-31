# contains the functions for training or evaluation
# 

import ml_collections
import torch
import torch.optim as optim
import torch.nn as nn

import losses

from NeuralNet import NeuralNetwork
# from AlexNet import AlexNet
from models import ncsnpp
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import sde_lib

import os
from utils import save_checkpoint,restore_checkpoint
import data_utils

from time import time


def get_configs():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 2
    training.epochs = 1000
    # training.n_iters = 130
    training.snapshot_freq = 50000
    training.log_freq = 25
    training.eval_freq = 25
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.sde = 'vesde'
    training.continuous = True

    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.075
    sampling.method = 'pc'
    sampling.predictor = 'reverse_diffusion'
    sampling.corrector = 'langevin'

    # Evaluation
    # config.eval = evaluate = ml_collections.ConfigDict()
    # evaluate.begin_ckpt = 50
    # evaluate.end_ckpt = 96
    # evaluate.batch_size = 8
    # evaluate.enable_sampling = True
    # evaluate.num_samples = 50000
    # evaluate.enable_loss = True
    # evaluate.enable_bpd = False
    # evaluate.bpf_dataset = 'test'

    # Data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'LSUN'
    # data.dataset = 'fastmri_knee'
    data.image_size = 256
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.num_channels = 1
    # data.root = ''
    data.is_multi = False
    data.is_complex = True
    data.nslices = 30

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 378
    model.sigma_min = 0.01
    model.num_scales = 2000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.
    model.embedding_type = 'fourier'
    model.name = 'ncsnpp'
    model.scale_by_sigma = True
    model.ema_rate = 0.999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1,2,2,2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1,3,3,1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'residual'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.fourier_scale = 16
    model.conv_size = 3

    # Optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    config.seed = 42
    # config.device = torch.device('cpu')
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    return config

def train_classification():
    '''Running training of the model
    for a classifier
    '''

    # Create directory for logs

    # Initialize the model
    cl_model = NeuralNetwork()
    print(cl_model)

    # Select optimizer
    optimizer = torch.optim.SGD(cl_model.parameters(),lr=1e-3)

    # Build data iterator
    train_ds,eval_ds = datasets.get_dataset('FashionMNIST')
    # train_iter = iter(train_ds)
    # eval_iter = iter(eval_ds)

    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # One-step training function
    def train_step_fn(model,batchdata,batchlabel):
        model.train()
        optimizer.zero_grad()
        pred = model(batchdata)
        loss = loss_fn(pred,batchlabel)
        loss.backward()
        optimizer.step()
        return loss

    # One-step evaluation function
    # def eval_step_fn(model,batchdata,batchlabel):
    #     model.eval()
    #     with torch.no_grad():
    #         pred = model(batchdata)
    #         correct = (pred.argmax(1)==y).type(torch.float).sum().item()
    #     return correct

    # Optimization
    initial_epoch = 0
    num_train_epoch = 5
    for epoch in range(initial_epoch,num_train_epoch+1):
        # databatch,labelbatch = next(train_iter)
        for step,(databatch,labelbatch) in enumerate(train_ds):
            # print(labelbatch.shape)
            loss = train_step_fn(cl_model,databatch,labelbatch)
            if step%100==0:
                print('[step={}][loss={}]'.format(step,loss))



def train():
    '''Run training of the model

    '''
    config = get_configs()
    # device = 'cpu'
    epoch = 0

    torch.manual_seed(config.seed)

    # Create directory for logs
    workdir = './training_log'
    if not os.path.exists(workdir): os.mkdir(workdir)
    log_fname = './training_log/logs.csv'
    if not os.path.exists(log_fname):
        with open(log_fname,'w') as f_log:
            f_log.write('time,epoch,step,loss,evalloss\n')


    # Initialize the model
    # df_model = NeuralNetwork() # TODO to implement the model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),decay=config.model.ema_rate)

    # Select optimizer
    optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999), 
                           eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, epoch=0)
    # print(state)
    
    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir,"checkpoints")
    checkpoint_meta_dir = os.path.join(workdir,"checkpoint_meta")
    if not os.path.exists(checkpoint_dir): os.mkdir(checkpoint_dir)
    if not os.path.exists(checkpoint_meta_dir): os.mkdir(checkpoint_meta_dir)
    # If there is intermediate checkpoints, and need to run from it
    if True:
        state = restore_checkpoint(os.path.join(checkpoint_meta_dir,"checkpoint_45.pth"),state,config.device)
        initial_step = int(state['step'])
        initial_epoch = int(state['epoch'])
        print(f'initial epoch = {initial_epoch}')
    else:
        initial_epoch = 0
    num_train_epoch = config.training.epochs

    # Build data iterator
    # train_ds,eval_ds = datasets.get_dataset('FashionMNIST')
    train_ds,eval_ds = datasets.get_dataset('fastmri_prostate',batch_size=config.training.batch_size)
    train_iter = iter(train_ds)
    # eval_iter = iter(eval_ds)

    # Create data normalizer and its inverse
    scaler = data_utils.get_data_scaler(config)
    inverse_scaler = data_utils.get_data_inverse_scaler(config)


    # Setup SDEs
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,sigma_max=config.model.sigma_max,N=config.model.num_scales)
    sampling_eps = 1e-3
    # print(sde)

    # Define the loss function
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    loss_fn = losses.get_sde_loss_fn(
        sde,train=True,reduce_mean=reduce_mean,continuous=True,likelihood_weighting=likelihood_weighting
    )


    # Prepare functions for training
    # One-step training function
    def train_step_fn(state, batchdata):
        model = state['model']
        optimizer = state['optimizer']
        optimizer.zero_grad()
        loss = loss_fn(model,batchdata)
        loss.backward()
        # may include other operation when optimizing
        optimizer.step()
        # 
        state['step'] += 1
        state['ema'].update(model.parameters())
        # Return of loss
        return loss
    # One-step evaulation function
    def eval_step_fn(state, batchdata):
        model = state['model']
        with torch.no_grad():
            ema = state['ema']
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            loss = loss_fn(model,batchdata)
            ema.restore(model.parameters())
        return loss

    # Build sampling functions
    # TODO



    start_time = time()

    # Training
    fcheckpoint = os.path.join(checkpoint_dir,'checkpoint_.pth')
    for epoch in range(initial_epoch, num_train_epoch):
        total_time = time() - start_time
        print(f'Epoch: {epoch} ============================ {total_time/60/60} h')
        for step,(databatch,labelbatch) in enumerate(train_ds):
            # Get the batch of data, and move to target device
            databatch = databatch.to(config.device)
            databatch = scaler(databatch)
            # ------------------------
            # print(databatch.shape)
            # print(labelbatch.shape)
            # print(databatch.device)
            # loss = loss_fn(score_model,databatch)

            
            # Execute one training step
            loss = train_step_fn(state,databatch)
            # print(loss)

            # Report the evaluation:
            if step % config.training.eval_freq  == 0:
                # print('step={}'.format(step))
                print('[step][{:6}] [loss][{}] --- [ {} h ]'.format(step,loss,(time()-start_time)/60/60))
                # print('databatch',torch.mean(databatch))
                # try:
                #     (eval_batch, evallabelbatch) = next(eval_iter)
                # except:
                #     eval_iter = iter(eval_ds)
                #     (eval_batch, evallabelbatch) = next(eval_iter)
                # eval_loss = eval_step_fn(score_model,eval_batch)
                # print('[step][{:6}] [loss][{}] [eval loss][{}]'.format(step,loss,eval_loss))
            
            # Save/Write into logs:
            if step % config.training.log_freq == 0:
                with open(log_fname,'a') as f_log:
                    # f_log.write('time,epoch,step,loss,evalloss')
                    logline = f'{time()-start_time},{epoch},{step},{loss},0\n'
                    f_log.write(logline)
                
            
            # xxxx a silly test
            # if step > 55:
            #     break
        # Save checkpoint for each epoch
        print('save check point')
        state = dict(optimizer=optimizer, model=score_model, ema=ema, step=step, epoch=epoch+1)
        if epoch%50 != 0:
            try:
                os.remove(fcheckpoint)
            except:
                pass
        fcheckpoint = os.path.join(checkpoint_dir,f'checkpoint_{epoch+1}.pth')
        save_checkpoint(fcheckpoint,state)

        # Generate samples if needed
        # TODO (not necessary now)

    return


def task_mri_recon():
    '''perform a mri reconstruction task'''
    return