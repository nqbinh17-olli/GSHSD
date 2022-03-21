import numpy as np
from decorators import timer, debug
import json
import random

def sigmoid(z):
    return 1/(1+np.e**(-z))

def timer_wrap(func, logit):
    return timer(func) if logit else func

def debug_wrap(func, logit):
    return debug(func) if logit else func

def cal_padding(insize, outsize, kernel_size, stride):
    padding = stride*(outsize-1) - insize + kernel_size
    return padding