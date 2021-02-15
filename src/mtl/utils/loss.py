# import statistics 
import numpy as np
import torch

__all__ = ['sumloss', 'avgloss', 'weightedavg', 'weightedsum', "weighted_lossp_sum", "weighted_lossp_avg"]

#-------sum loss function
def sumloss(head_losses):
    loss = 0
    for head in head_losses:
        loss += head
    return loss

#-------avg loss function
def avgloss(head_losses):
    loss = 0
    for head in head_losses:
        loss += head
    count = len(head_losses)
    return loss/count

#-------weighted loss avg function
# len of wieght loss should be the same as head count 
def weightedavg(head_losses, weights):
    loss = 0
    for head, weight in zip(head_losses, weights):
        loss += head*weight
    count = len(head_losses)
    return loss/count

#-------weighted loss sum function
# len of wieght loss should be the same as head count 
def weightedsum(head_losses, weights):
    loss = 0
    for head, weight in zip(head_losses, weights):
        loss += head*weight
    return loss

def weighted_lossp_sum(head_losses):
    loss = 0
    # m = statistics.mean(np.array(head_losses).astype(np.float32))
    m = torch.mean(head_losses)
    for head in head_losses:
        loss += ((head/m) * head)
    return loss

def weighted_lossp_avg(head_losses):
    loss = 0
    # m = statistics.mean(np.array(head_losses).astype(np.float32))
    m = torch.mean(head_losses)
    for head in head_losses:
        loss += ((head/m) * head)
    count = len(head_losses)
    return loss/count

