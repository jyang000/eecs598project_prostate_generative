# 

import torch
import matplotlib.pyplot as plt


import datasets





if __name__=='__main__':
    train_ds,eval_ds = datasets.get_dataset('fastmri_prostate')
    train_iter = iter(train_ds)

    for k in range(10):
        databatch,labalbatch = next(train_iter)
        print(databatch.shape)
        img = databatch.squeeze()
        fname = 'test/prostate_{}.png'.format(k)
        plt.imsave(fname,img,cmap='gray')