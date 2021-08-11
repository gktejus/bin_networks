import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import random
import pandas as pd

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

<<<<<<< HEAD
def adjust_learning_rate(optimizer, epoch, args):
=======
    def adjust_learning_rate(optimizer, epoch, args):
>>>>>>> f43c7ec87c42f076c98915f161b1317c36d030d8
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    if args.scheduler_type==1:
        lr = args.lr * (0.5 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        if epoch in [args.epochs*0.5, args.epochs*0.75]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1