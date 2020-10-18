
__all__ = ['sumloss', 'avgloss', 'weightedavg', 'weightedsum']

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

