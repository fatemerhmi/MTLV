import ast

__all__ = ['sumloss', 'avgloss', 'weightedloss']

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
def weightedloss(head_losses, weights):
    weights = ast.literal_eval(weights)
    loss = 0
    for head, weight in zip(head_loss, weights):
        loss = head*weight
    print(head_losses)
    print(type(head_loss))
    print(weights)
    print(type(weights))

    return loss

#-------
# def 