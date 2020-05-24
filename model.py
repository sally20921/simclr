import  os 
import torch  
from modules import SimCLR, LARS

def load_model(args, loader):
    model=SimCLR(args)
    model=model.to(args.device)

    learning_rate  =  0.3 * args.batch_size /256

