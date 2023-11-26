# datasets used for the project




import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor





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

def get_dataset(dataset_name):
    if dataset_name == 'FashionMNIST':
        train_ds,eval_ds = fashionmnist()
    elif dataset_name == 'MNIST':
        train_ds,eval_ds = mnist()
    elif dataset_name == 'fastmri_prostate':
        pass
    elif dataset_name == 'CIFAR10':
        pass
    else:
        print('dataset "{}" not supported'.format(dataset_name))
        exit(1)
    return train_ds, eval_ds

if __name__=='__main__':
    # a test
    train_ds,test_ds = get_dataset(dataset_name='FashionMNIST')
    train_ds_iter = iter(train_ds)
    x,label = next(train_ds_iter)
    print(x.shape)