"""
Computer Vision Dataset Loader

This module provides a unified interface for loading various computer vision datasets
including MNIST, FashionMNIST, CIFAR10, CIFAR100, ImageNet, and SVHN. It supports both
PyTorch's native DataLoader and FFCV (Fast Forward Computer Vision) for faster data loading.

Key features:
- Supports multiple popular CV datasets
- Automatic fallback from FFCV to PyTorch DataLoader
- Standardized data transformations and normalization

Author: Dong Wang (dong.wang@tugraz.at)
Date: 2024-01-30
"""

import os
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision import transforms, datasets
import numpy as np
import torch
import torchvision

dataset_infor = {
    'FashionMNIST':{'num_classes':10, 'num_channels':1},
    'MNIST':{'num_classes':10, 'num_channels':1},
    'ImageNet':{'num_classes':1000, 'num_channels':3},
    'CIFAR10':{'num_classes':10, 'num_channels':3},
    'CIFAR100':{'num_classes':100, 'num_channels':3}, 
    'CIFAR10_split_a':{'num_classes':10, 'num_channels':3},
    'CIFAR10_split_b':{'num_classes':10, 'num_channels':3},
    'SVHN':{'num_classes':10, 'num_channels':3},
    'SVHN_split_a':{'num_classes':10, 'num_channels':3},
    'SVHN_split_b':{'num_classes':10, 'num_channels':3},
}
def ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline):
    from ffcv.loader import Loader, OrderOption
    import os
    output_dir = './ffcv_datasets'
    # os.makedirs(output_dir, exist_ok=True)
    if split=="train":
        sub="train"
    else:
        sub="test"
    #todo: check file exists.
    output_file = f'{output_dir}/{dataset_name}_{sub}_ffcv.beton'

    data_loader = Loader(output_file, batch_size=batch_size, num_workers=num_workers, order=OrderOption.RANDOM if split=="train" else OrderOption.SEQUENTIAL, distributed=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        })
    return data_loader

def load_data(split, dataset_name, datadir, nchannels, batch_size,shuffle,device,num_workers=4):
    ## https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
    # todo: support `nchannels`
    if dataset_name in ['CIFAR10_split_a', 'CIFAR10_split_b']:
        get_dataset = getattr(datasets, "CIFAR10")
    elif dataset_name in ['SVHN_split_a', 'SVHN_split_b']:
        get_dataset = getattr(datasets, "SVHN")
    else:
        get_dataset = getattr(datasets, dataset_name)


    if dataset_name == 'MNIST':
        mean, std = [0.1307], [0.3081]
        normalize = transforms.Normalize(mean=mean, std=std)
        tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
        print("Using PyTorch dataset.")
    
    elif dataset_name == 'SVHN':
        mean=[0.4377, 0.4438, 0.4728]
        std=[0.1980, 0.2010, 0.1970]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert, Cutout
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")
        except ImportError:
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")

    elif dataset_name == 'CIFAR10':
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2470, 0.2435, 0.2616]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")
    elif dataset_name == 'CIFAR10_split_a':
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2470, 0.2435, 0.2616]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            split_label = 5
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
                np_target = np.array(dataset.targets)
                dataset.targets = np_target[np_target < split_label]
                dataset.targets = dataset.targets[dataset.targets < split_label]
                dataset.data = dataset.data[np_target < split_label]
            else:
                dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
                np_target = np.array(dataset.targets)
                dataset.targets = np_target[np_target < split_label]
                dataset.targets = dataset.targets[dataset.targets < split_label]
                dataset.data = dataset.data[np_target < split_label]
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")
    elif dataset_name == 'CIFAR10_split_b':
        mean=[0.4914, 0.4822, 0.4465]
        std=[0.2470, 0.2435, 0.2616]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            split_label = 4
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
                np_target = np.array(dataset.targets)
                dataset.targets = np_target[np_target > split_label]
                dataset.targets = dataset.targets[dataset.targets > split_label]
                dataset.data = dataset.data[np_target > split_label]
            else:
                dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
                np_target = np.array(dataset.targets)
                dataset.targets = np_target[np_target > split_label]
                dataset.targets = dataset.targets[dataset.targets > split_label]
                dataset.data = dataset.data[np_target > split_label]
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")
    elif dataset_name == 'SVHN_split_a':
        mean=[0.4377, 0.4438, 0.4728]
        std=[0.1980, 0.2010, 0.1970]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            split_label = 5
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
                np_target = np.array(dataset.labels)
                dataset.labels = np_target[np_target < split_label]
                dataset.labels = dataset.labels[dataset.labels < split_label]
                dataset.data = dataset.data[np_target < split_label]
            else:
                dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
                np_target = np.array(dataset.labels)
                dataset.labels = np_target[np_target < split_label]
                dataset.labels = dataset.labels[dataset.labels < split_label]
                dataset.data = dataset.data[np_target < split_label]
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")
    elif dataset_name == 'SVHN_split_b':
        mean=[0.4377, 0.4438, 0.4728]
        std=[0.1980, 0.2010, 0.1970]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            split_label = 4
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
                np_target = np.array(dataset.labels)
                dataset.labels = np_target[np_target > split_label]
                dataset.labels = dataset.labels[dataset.labels > split_label]
                dataset.data = dataset.data[np_target > split_label]
            else:
                dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
                np_target = np.array(dataset.labels)
                dataset.labels = np_target[np_target > split_label]
                dataset.labels = dataset.labels[dataset.labels > split_label]
                dataset.data = dataset.data[np_target > split_label]
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")
    elif dataset_name == 'CIFAR100':
        mean=[0.5071, 0.4865, 0.4409]
        std=[0.2673, 0.2564, 0.2762]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert, Cutout
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            normalize = transforms.Normalize(mean=mean, std=std)
            # https://github.com/pytorch/examples/blob/a38cbfc6f817d9015fc67a6309d3a0be9ff94ab6/imagenet/main.py#L239-L255
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)           
            print("Using PyTorch dataset.")
            
    elif dataset_name == 'ImageNet':
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        try:
            import ffcv
            from ffcv.fields import IntField, RGBImageField
            from ffcv.writer import DatasetWriter
            from ffcv.transforms import RandomHorizontalFlip, RandomTranslate, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert, Cutout
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [RandomResizedCropRGBImageDecoder((224, 224)), 
                                RandomHorizontalFlip(),
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                NormalizeImage(np.array(mean)*255, np.array(std)*255, np.float32),
                                ]
            elif split == 'test':
                image_pipeline =[
                    CenterCropRGBImageDecoder((224,224), ratio=224/256),
                    ToTensor(),
                    ToDevice(device, non_blocking=True),
                    ToTorchImage(),
                    NormalizeImage(np.array(mean)*255, np.array(std)*255, np.float32),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            normalize = transforms.Normalize(mean=mean, std=std)
            # https://github.com/pytorch/examples/blob/a38cbfc6f817d9015fc67a6309d3a0be9ff94ab6/imagenet/main.py#L239-L255
            tr_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, split='train', transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, split='val', transform=val_transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")

    elif dataset_name == 'FashionMNIST':
        mean=[0.5]
        std=[0.5]
        normalize = transforms.Normalize(mean=mean, std=std)
        tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
        if split == 'train':
            dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
        else:
            dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
        print("Using PyTorch dataset.")
    elif dataset_name == 'SVHN':
        mean=[0.4377, 0.4438, 0.4728]
        std=[0.1980, 0.2010, 0.1970]
        try:
            import ffcv
            from ffcv.transforms import RandomHorizontalFlip, NormalizeImage, Squeeze,  RandomHorizontalFlip, ToTorchImage, ToDevice, Convert, ToTensor, Convert
            from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder,SimpleRGBImageDecoder
            from ffcv.fields.basics import IntDecoder
            label_pipeline = [
                IntDecoder(),
                ToTensor(),
                Squeeze(),
                ToDevice(device, non_blocking=True),

            ]
            if split =='train':
                image_pipeline= [SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), 
                                # transforms.RandomRotation(15),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                                ]
            elif split == 'test':
                image_pipeline =[SimpleRGBImageDecoder(), 
                                ToTensor(),
                                ToDevice(device, non_blocking=True),
                                ToTorchImage(),
                                Convert(torch.float),
                                torchvision.transforms.Normalize(np.array(mean)*255, np.array(std)*255),
                ]
            data_loader = ffcv_data(dataset_name, split, num_workers, batch_size, image_pipeline, label_pipeline)
            print("Using FFCV dataset.")

        except ImportError:
            normalize = transforms.Normalize(mean=mean, std=std)
            tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
            val_transform = transforms.Compose([transforms.ToTensor(), normalize])
            if split == 'train':
                dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
            else:
                dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
            print("Using PyTorch dataset.")
    else:
        raise NotImplementedError(f"Non-supported dataset")

    # normalize = transforms.Normalize(mean=mean, std=std)
    # if dataset_name == 'ImageNet':
    #     # https://github.com/pytorch/examples/blob/a38cbfc6f817d9015fc67a6309d3a0be9ff94ab6/imagenet/main.py#L239-L255
    #     tr_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    #     val_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    # elif dataset_name == "CIFAR10":
    #     tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    #     val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    # elif dataset_name == "CIFAR100":
    #     tr_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(), transforms.RandomRotation(15), transforms.ToTensor(), normalize])
    #     val_transform = transforms.Compose([transforms.ToTensor(), normalize])
    # elif dataset_name == 'FashionMNIST':
    #     tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    #     val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    # else:
    #     tr_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])
    #     val_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize])


    # if dataset_name == 'SVHN':
    #     if split == 'train':
    #         dataset = get_dataset(root=datadir, split='train', download=True, transform=tr_transform)
    #     else:
    #         dataset = get_dataset(root=datadir, split='test', download=True, transform=val_transform)
    # elif dataset_name== 'ImageNet':
    #     if split == 'train':
    #         dataset = get_dataset(root=datadir, split='train', transform=tr_transform)
    #     else:
    #         dataset = get_dataset(root=datadir, split='val', transform=val_transform)
    # else:
    #     if split == 'train':
    #         dataset = get_dataset(root=datadir, train=True, download=True, transform=tr_transform)
    #     else:
    #         dataset = get_dataset(root=datadir, train=False, download=True, transform=val_transform)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,num_workers=4)



    return data_loader