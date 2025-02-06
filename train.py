"""
Neural Network Model Training Script
===================================

This script provides a comprehensive training pipeline for various neural network architectures
on different datasets (CIFAR10, CIFAR100, FashionMNIST, etc.). It includes functionality for:
- Model training and evaluation
- Learning rate scheduling with warmup
- Logging metrics to Weights & Biases
- Saving model checkpoints
- Support for different model architectures (ResNet, VGG, LeNet, etc.)
- Configurable model width scaling

Author: Dong Wang (dong.wang@tugraz.at)
Date: 2024-01-30

Usage examples are provided at the bottom of the script for training different models
on various datasets with customizable hyperparameters.
"""

import torch
from torch.cuda.amp import autocast
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import wandb
import argparse
import models
from models import *
from cvdataset import load_data, dataset_infor
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler
torch.cuda.empty_cache()

def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in tqdm(train_loader, "Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # with autocast(dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        # L1 regularization
        # l1_lambda = 5e-4
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        # loss+=l1_lambda * l1_norm
        # L1 regularization

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()


    train_accuracy = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
    return train_accuracy/100, avg_train_loss


def test(test_loader, model, criterion, epoch, device):
    model.eval()

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
      for inputs, labels in test_loader:
      # for inputs, labels in tqdm(test_loader, "Testing"):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            # L1 regularization
            # l1_lambda = 5e-4
            # l1_norm = sum(p.abs().sum() for p in model.parameters())
            # loss+=l1_lambda * l1_norm
            # L1 regularization

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()


      test_accuracy = 100 * test_correct / test_total
      avg_test_loss = test_loss / len(test_loader)
      print(f"Epoch {epoch+1}, Testing Loss: {avg_test_loss:.4f}, Testing Accuracy: {test_accuracy:.2f}%")
      return test_accuracy/100, avg_test_loss
    
# evaluate the number of accurate examples
def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    with torch.no_grad(), autocast():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            pred = outputs.argmax(dim=1)
            correct += (labels.to(device) == pred).sum().item()
    return correct

# evaluates acc and loss on a dataset.
def evaluate2(model, loader, device='cpu', verbose=False):
    model.eval()
    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            pred = outputs.argmax(dim=1)
            correct += (labels.to(device) == pred).sum().item()
            total += len(labels)
            loss = F.cross_entropy(outputs, labels.to(device))
            losses.append(loss.item())
    if verbose:
        print(f'Accuracy: {100*correct / total:.4}% | Loss: {np.array(losses).mean():.4}')
    return 100* correct / total, np.array(losses).mean()

# Evaluate model on both training dataset and test dataset
def full_eval(model, train_loader, test_loader, device='cpu'):
    tr_acc, tr_loss = evaluate2(model, loader=train_loader, device=device, verbose=False)
    te_acc, te_loss = evaluate2(model, loader=test_loader, device=device, verbose=False)
    print(f'training accuracy: {100*tr_acc:.4}% | training loss: {tr_loss:.4} | testing accuracy: {100*te_acc:.4}% | testing loss: {te_loss:.4}')
    return (100*tr_acc, tr_loss, 100*te_acc, te_loss)


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super(WarmupLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        return self.base_lrs



def main():
    model_names = sorted(name for name in models.__dict__
                     if name.islower()
                     and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--dataset', default='FashionMNIST', type=str, help='FashionMNIST')
    parser.add_argument('--datadir', default='datasets', type=str)
    parser.add_argument('--batch_size', default=256 , type=int)
    parser.add_argument('--save-dir', dest='save_dir', default='save_temp', type=str)
    parser.add_argument('--model', '-m', metavar='MODEL', default='lenet5',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: lenet5)')
    parser.add_argument('--wider_factor', default=1, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--gpus', default=0, type=int)
    parser.add_argument('--output', default='./weights', type=str)
    args = parser.parse_args()

    if args.gpus==-1:
        device= torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else torch.device('cpu') )
    print(f'using device {device}')
    # torch.cuda.set_device(device)
    # print(f'using device {device}')
    train_loader = load_data("train", args.dataset, datadir=args.datadir, nchannels=dataset_infor[args.dataset]['num_channels'], batch_size=args.batch_size, shuffle=True,device=device,num_workers=4)
    test_loader = load_data("test", args.dataset, datadir=args.datadir, nchannels=dataset_infor[args.dataset]['num_channels'], batch_size=args.batch_size, shuffle=True,device=device,num_workers=4)

    # model_profile_str = f'{args.model}_{args.wider_factor}Xwider'
    model_profile_str = f'{args.model}_{args.wider_factor}Xwider_{args.dataset}' if args.wider_factor > 1 else f'{args.model}_{args.dataset}'
    print(f'trainig {model_profile_str} {args.epochs} epochs with lr={args.learning_rate} | batchsize={args.batch_size} on {args.dataset}and will save weights in {args.output}')
    wandb_proiject_name = "model folding" 
    experiment_name = f"train_{model_profile_str}"
    run = wandb.init(project=wandb_proiject_name, name=experiment_name, entity="naguoyu", 
               config={"dataset":args.dataset},
               )
    model = models.__dict__[args.model](
                                        # pretrained=True,
                                        # weights=ResNet34_Weights.IMAGENET1K_V1
                                        wider_factor=args.wider_factor,
                                        n_channels = dataset_infor[args.dataset]['num_channels'],
                                        num_classes= dataset_infor[args.dataset]['num_classes'],
                                        )
    # for param in model.parameters():
    #     param.requires_grad = False
    model.to(device)
    print(model)
    epochs = args.epochs

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs*1), eta_min=0)
    warmup_epochs = 2
    scheduler_warmup = WarmupLR(optimizer, warmup_epochs=warmup_epochs)

    criterion = nn.CrossEntropyLoss().to(device)
    print('Starting training...')

    for epoch in range(epochs):
        train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch, device=device)
        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler.step()
        test_acc, test_loss = test(test_loader, model, criterion, epoch, device=device)
        run.log({"train/acc": train_acc, "train/loss": train_loss, "test/acc": test_acc, "test/loss": test_loss})

    test_acc = test(test_loader, model, criterion, epoch, device=device)

    print(f'Test accuracy: {test_acc}')
    print('\n')


    ######## write to the bucket
    destination_name = f'{args.output}'
    os.makedirs(destination_name, exist_ok=True)
    filename = f'{destination_name}/{model_profile_str}.pth'   
    torch.save(model.state_dict(), filename)
    print(f'Models are saved in {filename}')
    run.finish()
if __name__ == '__main__':
    main()
# You can use the following commands to train the models. With --wider_factor to train the wider model, you can control the width of the model.

# Training resnet18 on CIFAR10
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=CIFAR10 --model=resnet18 --batch_size=128 --epochs=100 --learning_rate=0.1 --wider_factor=1
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=CIFAR10_split_a --model=resnet18 --batch_size=128 --epochs=100 --learning_rate=0.1 --wider_factor=1
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=CIFAR10_split_b --model=resnet18 --batch_size=128 --epochs=100 --learning_rate=0.1 --wider_factor=1


# Training resnet18 on SVHN
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=SVHN --model=resnet18 --batch_size=256 --epochs=50 --learning_rate=0.01 --wider_factor=1
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=SVHN_split_a --model=resnet18 --batch_size=256 --epochs=50 --learning_rate=0.01 --wider_factor=1
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=SVHN_split_b --model=resnet18 --batch_size=256 --epochs=50 --learning_rate=0.01 --wider_factor=1

# Training vgg11_bn on CIFAR10
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=CIFAR10 --model=vgg11_bn --epochs=100 --learning_rate=0.1 --wider_factor=1
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=CIFAR10_split_a --model=vgg11_bn --epochs=100 --learning_rate=0.1 --wider_factor=1
# CUDA_VISIBLE_DEVICES=6 python train.py --dataset=CIFAR10_split_b --model=vgg11_bn --epochs=100 --learning_rate=0.1 --wider_factor=1


# Training resnet50 on CIFAR100
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset=CIFAR100 --model=resnet50 --batch_size=512 --epochs=100 --learning_rate=0.1 --wider_factor=1

# Imagenet
# You do not need to train models on Imagenet, because torchvision has already provided the pretrained weights.
