import os
import torch
import torchvision
import  argparse 


from modules import NT_Xent
from modules import Augmentation
from model import load_model, save_model

def train(args,  train_loader, model, criterion, optimizer, writer):

