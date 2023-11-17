# contains the functions for training or evaluation
# 

import ml_collections
import torch
import torch.optim as optim

import losses

from models import NeuralNetwork
import datasets


def get_configs():
    config = ml_collections.ConfigDict()


    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 130
    training.snapshot_freq = 50000
    training.log_freq = 50
    training.eval_freq = 20
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    
    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    return config



def train():
    '''Run training of the model

    '''
    config = get_configs()

    # Create directory for logs


    # Initialize the model
    df_model = NeuralNetwork() # TODO to implement the model
    optimizer = optim.Adam(df_model.parameters(), lr=config.optim.lr, betas=(config.optim.beta1, 0.999), 
                           eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
    state = dict(optimizer=optimizer, model=df_model, step=0)
    print(state)
    
    # Create checkpoints directory

    # Build data iterator
    train_ds,eval_ds = datasets.get_dataset('FashionMNIST')
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Define the loss function
    def loss_fn(model, batchdata):
        # use the DDPM loss function
        return 0


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



    initial_step = 0
    num_train_step = config.training.n_iters
    # Training
    for step in range(initial_step, num_train_step+1):

        # Get the batch of data
        databatch, labelbatch = next(train_iter)
        # print(databatch.shape)
        # print(labelbatch.shape)

        # Execute one training step
        # loss = train_step_fn(df_model,databatch)

        # Report the evaluation:
        if step % config.training.eval_freq  == 0:
            print('step={}'.format(step))
            # eval_batch = None
            # eval_loss = eval_step_fn(df_model,eval_batch)

        # Save checkpoint and generate samples if needed
        # TODO (not necessary now)

    return


def task():
    '''perform a mri reconstruction task'''
    return