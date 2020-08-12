import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, BCELoss, CrossEntropyLoss


__all__ = ["HeadMultilabelCLS", "HeadMulticlassCLS"]

class Head():
    def __init__(self,hparams):
        self.inputLayer = hparams['inputLayer']
        self.run()
    def run(self):
        pass

class HeadMultilabelCLS(Head):
    def __init__(self,hparams):
        self.labels = hparams['labels'] # batch labels, or all labels
        self.device = hparams['device']
        self.num_labels = hparams['num_labels'] # number of labels
        self.taskspecificLayer = nn.Linear(hparams['input_size'], hparams['num_labels']).to(self.device) #classifier
        self.loss = None
        self.logits = None
        self.pred_labels = None
        
        super().__init__(hparams)
        
    def run(self):
        self.logits = self.taskspecificLayer(self.inputLayer)
        self.pred_label = torch.sigmoid(self.logits)
        
        loss_func = BCEWithLogitsLoss() 
        self.loss = loss_func(self.logits.view(-1,self.num_labels),
                         self.labels.type_as(self.logits).view(-1,self.num_labels)) #convert labels to float for calculation
        # loss_func = BCELoss() 
        # loss = loss_func(torch.sigmoid(logits.view(-1,num_labels)),b_labels.type_as(logits).view(-1,num_labels)) #convert labels to float for calculation
        
        return self.loss  
    
    
class HeadMulticlassCLS(Head):
    def __init__(self,hparams):
        super().__init__(hparams)

    def run(self):
        logits = self.taskspecificLayer(self.inputLayer)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
    
    
class HeadBinaryCLS(Head):
    def __init__(self,hparams):
        super().__init__(hparams)

    def run(self):
        pass