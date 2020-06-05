import os
import torch
import torchvision
import  argparse 


from modules import NT_Xent
from modules import Augmentation
from model import load_model, save_model

def train(args,  train_loader, model, criterion, optimizer):
    loss_epoch = 0

    return loss_epoch

def main(_run, _log):
    args = argparse.Namespace(**_run.config)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root = "./datasets"

    train_dataset = torchvision.datasets.CIFAR10(
            root, download=True, transform=Augmentation(size=32)
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    
    )

    model, optimizer = load_model(args, train_loader)
    
    criterion = NT_Xent(args.batch_size, args.temperature, args.device)

    save_model(args, model, optimizer)



