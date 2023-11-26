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


def get_configs():
    config = ml_collections.ConfigDict()

    # Training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 1
    training.epochs = 1000
    # training.n_iters = 130
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 100
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
    data.image_size = 256
    data.random_flip = True
    data.uniform_dequantization = False
    data.centered = False
    data.num_channels = 1
    # data.dataset = 'fastmri_knee'
    # data.root = ''
    # data.image_size = 320
    data.is_multi = False
    data.is_complex = True

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
    device = 'cpu'

    # Create directory for logs
    # TBD


    # Initialize the model
    # df_model = NeuralNetwork() # TODO to implement the model
    score_model = mutils.create_model(config)
    ema = ExponentialMovingAverage(score_model.parameters(),decay=config.model.ema_rate)

    # Select optimizer
    optimizer = optim.Adam(score_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999), 
                           eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
    state = dict(optimizer=optimizer, model=score_model, step=0)
    # print(state)
    
    # Create checkpoints directory

    # Build data iterator
    train_ds,eval_ds = datasets.get_dataset('FashionMNIST')
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Setup SDEs
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min,sigma_max=config.model.sigma_max,N=config.model.num_scales)
    sampling_eps = 1e-3
    # print(sde)

    # Define the loss function
    reduce_mean = config.training.reduce_mean
    likelihood_weighting = config.training.likelihood_weighting
    loss_fn = losses.get_sde_loss_fn(sde,train=True,reduce_mean=reduce_mean,continuous=True,likelihood_weighting=likelihood_weighting)


    # Prepare functions for training
    # One-step training function
    def train_step_fn(model, batchdata):
        optimizer.zero_grad()
        loss = loss_fn(model,batchdata)
        loss.backward()
        
        # may include other operation when optimizing
        optimizer.step()

        return loss
    # One-step evaulation function
    def eval_step_fn(model, batchdata):
        with torch.no_grad():
            loss = loss_fn(model,batchdata)
        return loss

    # Build sampling functions
    # TODO



    initial_step = 0
    num_train_step = 100
    # Training
    for step in range(initial_step, num_train_step+1):

        # Get the batch of data
        databatch, labelbatch = next(train_iter)
        # print(databatch.shape)
        # print(labelbatch.shape)

        # Execute one training step
        # loss = train_step_fn(score_model,databatch)

        # Report the evaluation:
        if step % config.training.eval_freq  == 0:
            print('step={}'.format(step))
            # eval_batch = None
            # eval_loss = eval_step_fn(score_model,eval_batch)

        # Save checkpoint and generate samples if needed
        # TODO (not necessary now)

    return


def task_mri_recon():
    '''perform a mri reconstruction task'''
    return