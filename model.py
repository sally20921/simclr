import  os 
import torch  
from modules import SimCLR, LARS

def load_model(args, loader):
    model=SimCLR(args)
    model=model.to(args.device)

    learning_rate  =  0.3 * args.batch_size /256
    optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "bias"],
        )

    return model, optimizer

def save_model(args, model, optimizer):
    out  = os.path.join(args.out_dir, "checkpoint_{}.tar".format(args.current_epoch))
    torch.save(model.state_dict(), out)

