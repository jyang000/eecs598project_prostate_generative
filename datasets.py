# datasets used for the project




import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
from pathlib import Path



# Some default configuration of the datasets
main_data_folder = '/scratch/Datasets/'


# Fashion MINST dataset -------------------- 
# for test purpose
def fashionmnist():
    '''
    default batch size=64
    data size=1*28*28
    '''
    training_data = datasets.FashionMNIST(
        root=main_data_folder+"FashionMNIST",
        train=True,
        download=False,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=main_data_folder+"FashionMNIST",
        train=False,
        download=False,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    # # train_dataloader()
    # print(type(training_data[0]), training_data[0][0].shape, training_data[0][1])

    # train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")
    return train_dataloader, test_dataloader

def mnist():
    '''
    MNIST dataset
    data size=1*28*28
    '''
    training_data = datasets.MNIST(
        root=main_data_folder+'MNIST',
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_data = datasets.MNIST(
        root=main_data_folder+'MNIST',
        train=False,
        download=False,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data, batch_size=32)
    test_dataloader = DataLoader(test_data, batch_size=32)

    return train_dataloader,test_dataloader


# -----------------------------------------


class Fastmri_Prostate(Dataset):
    def __init__(self, root, sort=True, transform=None, label_transform=None):
        self.root = root
        self.data_list = list(root.glob('*/*.npy'))
        self.transform = transform
        self.label_transform = label_transform
        if sort:
            self.data_list = sorted(self.data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        fname = self.data_list[idx]
        # data = np.load(fname).astype(np.complex64)
        data = np.load(fname)
        # print(data.shape)
        data = data[10:(10+256),32:(32+256)]
        data = np.expand_dims(data, axis=0)
        if self.transform:
            data = self.transform(data)
        data = data*3.3e3
        return data, str(fname)



def fastmri_prostate(batch_size=1):
    '''
    fastmri prostate dataset
    data size = 30*320*320
    '''
    train_dataset = Fastmri_Prostate(root=Path('/scratch/Datasets/fastmri/prostate_training_T2'))
    # print(train_dataset.__len__())
    # print(train_dataset.data_list)
    # return 0,0

    # eval_dataset = Fastmri_Prostate(root=Path('/scratch/Datasets/fastmri/prostate_validation_T2'))
    eval_dataset = None
    
    print('dataset size | train:',len(train_dataset.data_list), '| eval:',
        #   len(eval_dataset.data_list)
          )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    # eval_dataloader = DataLoader(
    #     dataset=eval_dataset,
    #     batch_size=4,
    #     shuffle=True,
    #     drop_last=True,
    # )
    eval_dataloader = None
    return train_dataloader,eval_dataloader


##########################################################
# Function that get the dataset

def get_dataset(dataset_name,batch_size=1):
    if dataset_name == 'FashionMNIST':
        train_ds,eval_ds = fashionmnist()
    elif dataset_name == 'MNIST':
        train_ds,eval_ds = mnist()
    elif dataset_name == 'fastmri_prostate':
        train_ds,eval_ds = fastmri_prostate(batch_size=batch_size)
    elif dataset_name == 'CIFAR10':
        pass
    else:
        print('dataset "{}" not supported'.format(dataset_name))
        exit(1)
    return train_ds, eval_ds

if __name__=='__main__':
    # a test
    train_ds,test_ds = get_dataset(dataset_name='fastmri_prostate')
    train_ds_iter = iter(train_ds)
    x,label = next(train_ds_iter)
    print(x.shape)
    print(label)
    print('data:',torch.mean(x),torch.var(x))

    xmaxlist = []
    xminlist = []
    for n,(x,label) in enumerate(train_ds):
        xmax = torch.max(x)
        xmin = torch.min(x)
        xmaxlist.append(xmax)
        xminlist.append(xmin)
    print(torch.max(torch.tensor(xmaxlist)),torch.min(torch.tensor(xmaxlist)))
    print(torch.max(torch.tensor(xminlist)))