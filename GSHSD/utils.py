import numpy as np
from decorators import timer, debug
import torch
import random, os

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(42)

def get_device():
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    else:
        device = torch.device("cpu")
    print(f"Using {device}")
    return device


def sigmoid(z):
    return 1/(1+np.e**(-z))

def timer_wrap(func, logit):
    return timer(func) if logit else func

def debug_wrap(func, logit):
    return debug(func) if logit else func

def cal_padding(insize, outsize, kernel_size, stride):
    padding = stride*(outsize-1) - insize + kernel_size
    return padding