# datasets used for the project




import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor







# Fashion MINST dataset -------------------- 
# for test purpose
def fashionmnist():
    '''
    default batch size=64
    '''
    training_data = datasets.FashionMNIST(
        root="/scratch/Datasets/FashionMNIST",
        train=True,
        download=False,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="/scratch/Datasets/FashionMNIST",
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




# -----------------------------------------

def get_dataset(dataset_name):
    if dataset_name == 'FashionMNIST':
        train_ds,eval_ds = fashionmnist()
        pass
    elif dataset_name == 'fastmri_prostate':
        pass
    elif dataset_name == 'CIFAR10':
        pass
    else:
        print('dataset "{}" not supported'.format(dataset_name))
        exit(1)
    return train_ds, eval_ds