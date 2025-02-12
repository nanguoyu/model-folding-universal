"""
FFCV Dataset Preparation Script
------------------------------

This script converts various image datasets (CIFAR10, CIFAR100, ImageNet) into FFCV format
for faster data loading during training. FFCV (Fast Forward Computer Vision) is a data loading
system that significantly accelerates training of deep learning models.

Supported datasets:
- CIFAR10 (including split variants A/B)
- CIFAR100
- ImageNet
- MNIST
- FashionMNIST
- SVHN

Author: Dong Wang (dong.wang@tugraz.at)
Date: 2024-01-30

Note: you should use the commands at the bottom of the script to prepare the dataset, just once.
"""

import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms, datasets
import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from cvdataset import balance_svhn_dataset, balance_gtsrb_dataset
import matplotlib.pyplot as plt
from collections import Counter

dataset_infor = {
    'FashionMNIST':{'num_classes':10, 'num_channels':1},
    'MNIST':{'num_classes':10, 'num_channels':1},
    'ImageNet':{'num_classes':1000, 'num_channels':3},
    'CIFAR10':{'num_classes':10, 'num_channels':3},
    'CIFAR100':{'num_classes':100, 'num_channels':3},
    'CIFAR10_split_a':{'num_classes':10, 'num_channels':3},
    'CIFAR10_split_b':{'num_classes':10, 'num_channels':3},
    'CIFAR100_split_a':{'num_classes':100, 'num_channels':3},
    'CIFAR100_split_b':{'num_classes':100, 'num_channels':3},
    'SVHN':{'num_classes':10, 'num_channels':3},
    'SVHN_split_a':{'num_classes':10, 'num_channels':3},
    'SVHN_split_b':{'num_classes':10, 'num_channels':3},
    'GTSRB':{'num_classes':43, 'num_channels':3},
    'GTSRB_split_a':{'num_classes':43, 'num_channels':3},
    'GTSRB_split_b':{'num_classes':43, 'num_channels':3},
}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

def prepare_data(split, dataset_name, datadir):
    import ffcv
    from ffcv.fields import IntField, RGBImageField
    from ffcv.writer import DatasetWriter
    if dataset_name in ['CIFAR10_split_a', 'CIFAR10_split_b']:
        get_dataset = getattr(datasets, "CIFAR10")
    elif dataset_name in ['CIFAR100_split_a', 'CIFAR100_split_b']:
        get_dataset = getattr(datasets, "CIFAR100")
    elif dataset_name in ['SVHN_split_a', 'SVHN_split_b']:
        get_dataset = getattr(datasets, "SVHN")
    elif dataset_name in ['GTSRB_split_a', 'GTSRB_split_b']:
        get_dataset = getattr(datasets, "GTSRB")
    else:
        get_dataset = getattr(datasets, dataset_name)

    import os
    output_dir = './ffcv_datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    if split=="train":
        sub="train"
    else:
        sub="test"

    output_file = f'{output_dir}/{dataset_name}_{sub}_ffcv.beton'
    if dataset_name == 'GTSRB':
        transform = transforms.Compose([transforms.Resize((32, 32))])
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=transform)
    elif dataset_name == 'GTSRB_split_a':
        transform = transforms.Compose([transforms.Resize((32, 32))])
        split_label = 22
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=transform)
        dataset = balance_gtsrb_dataset(dataset)
        filtered_samples = [(path, target) for path, target in dataset._samples if target < split_label]
        dataset._samples = filtered_samples

    elif dataset_name == 'GTSRB_split_b':
        transform = transforms.Compose([transforms.Resize((32, 32))])
        split_label = 21
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True, transform=transform)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True, transform=transform)
        dataset = balance_gtsrb_dataset(dataset)
        filtered_samples = [(path, target) for path, target in dataset._samples if target > split_label]
        dataset._samples = filtered_samples

    elif dataset_name == 'SVHN':
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True)
    elif dataset_name== 'ImageNet':
        image_field = RGBImageField(write_mode='smart', max_resolution=256, jpeg_quality=90)
        if split == 'train': 
            dataset = get_dataset(root=datadir, split='train')
        else:
            dataset = get_dataset(root=datadir, split='val')
    elif dataset_name == 'CIFAR10_split_a':
        split_label = 5
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target < split_label]
        dataset.targets = dataset.targets[dataset.targets < split_label]
        dataset.data = dataset.data[np_target < split_label]
    elif dataset_name == 'CIFAR10_split_b':
        split_label = 4
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target > split_label]
        dataset.targets = dataset.targets[dataset.targets > split_label]
        dataset.data = dataset.data[np_target > split_label]
    elif dataset_name == 'CIFAR100_split_a':
        split_label = 50
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target < split_label]
        dataset.targets = dataset.targets[dataset.targets < split_label]
        dataset.data = dataset.data[np_target < split_label]
    elif dataset_name == 'CIFAR100_split_b':
        split_label = 49
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True)
        np_target = np.array(dataset.targets)
        dataset.targets = np_target[np_target > split_label]
        dataset.targets = dataset.targets[dataset.targets > split_label]
        dataset.data = dataset.data[np_target > split_label]
    elif dataset_name == 'SVHN_split_a':
        split_label = 5
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True)
            dataset = balance_svhn_dataset(dataset)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True)
        dataset = balance_svhn_dataset(dataset)
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target < split_label]
        dataset.labels = dataset.labels[dataset.labels < split_label]
        dataset.data = dataset.data[np_target < split_label]
    elif dataset_name == 'SVHN_split_b':
        split_label = 4
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, split='train', download=True)
        else:
            dataset = get_dataset(root=datadir, split='test', download=True)
        dataset = balance_svhn_dataset(dataset)
        np_target = np.array(dataset.labels)
        dataset.labels = np_target[np_target > split_label]
        dataset.labels = dataset.labels[dataset.labels > split_label]
        dataset.data = dataset.data[np_target > split_label]
    else:
        image_field = RGBImageField(write_mode='smart', max_resolution=32, jpeg_quality=90)
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True)

    write_config = {
        'image': image_field,
        'label': IntField()
    }

    writer = DatasetWriter(output_file, write_config)
    writer.from_indexed_dataset(dataset)

# Now you can generate FFCV dataset before use it for training.

# CIFAT10
# prepare_data(split="train", dataset_name="CIFAR10", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR10", datadir="datasets")

# # CIFAR10_split_a
# prepare_data(split="train", dataset_name="CIFAR10_split_a", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR10_split_a", datadir="datasets")

# # CIFAR10_split_b
# prepare_data(split="train", dataset_name="CIFAR10_split_b", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR10_split_b", datadir="datasets")

# # CIFAR100
# prepare_data(split="train", dataset_name="CIFAR100", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR100", datadir="datasets")

# # For ImageNet, `~/data/ImageNet` should be a folder containing files ILSVRC2012_devkit_t12.tar.gz, ILSVRC2012_img_train.tar, ILSVRC2012_img_val.tar 
# prepare_data(split="train", dataset_name="ImageNet", datadir="~/data/ImageNet")
# prepare_data(split="test", dataset_name="ImageNet", datadir="~/data/ImageNet")

# #SVHN
# prepare_data(split="train", dataset_name="SVHN", datadir="datasets")
# prepare_data(split="test", dataset_name="SVHN", datadir="datasets")

# #SVHN_split_a
# prepare_data(split="train", dataset_name="SVHN_split_a", datadir="datasets")
# prepare_data(split="test", dataset_name="SVHN_split_a", datadir="datasets")

# #SVHN_split_b
# prepare_data(split="train", dataset_name="SVHN_split_b", datadir="datasets")
# prepare_data(split="test", dataset_name="SVHN_split_b", datadir="datasets")

#CIFAR100_split_a
# prepare_data(split="train", dataset_name="CIFAR100_split_a", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR100_split_a", datadir="datasets")

#CIFAR100_split_b
# prepare_data(split="train", dataset_name="CIFAR100_split_b", datadir="datasets")
# prepare_data(split="test", dataset_name="CIFAR100_split_b", datadir="datasets")

# GTSRB
# prepare_data(split="train", dataset_name="GTSRB", datadir="datasets")
# prepare_data(split="test", dataset_name="GTSRB", datadir="datasets")

# GTSRB_split_a
prepare_data(split="train", dataset_name="GTSRB_split_a", datadir="datasets")
prepare_data(split="test", dataset_name="GTSRB_split_a", datadir="datasets")

# GTSRB_split_b
prepare_data(split="train", dataset_name="GTSRB_split_b", datadir="datasets")
prepare_data(split="test", dataset_name="GTSRB_split_b", datadir="datasets")
