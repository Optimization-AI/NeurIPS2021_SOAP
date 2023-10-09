import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sigmoidf = nn.Sigmoid()


def squared_hinge_loss(predScore, targets, b):


    squared_hinge = (1-targets*(predScore - b))
    squared_hinge[squared_hinge <=0] = 0


    return squared_hinge ** 2


def sigmoid_loss(pos, neg, beta=2.0):
    return 1.0 / (1.0 + torch.exp(beta * (pos - neg)))

def logistic_loss(pos, neg, beta = 1):
    return -torch.log(1/(1+torch.exp(-beta * (pos - neg))))
