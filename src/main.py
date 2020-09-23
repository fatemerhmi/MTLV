import pandas as pd
import ast
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import click

from mtl.models.bert import BertCLS
from mtl.utils.evaluate import *
from mtl.heads.utils import padding_heads, group_heads
from mtl.heads.clsHeads import *
import mtl.utils.configuration as configuration
from training import train



@click.group()
def main():
    """The package help is as follows."""
    pass

@main.command("run")
@click.option('--config', '-cfg', type=click.Path(exists=True, resolve_path=True), help='Configuration file.')
@click.option('--gpu-id', '-g', type=click.INT, default=0, help='GPU ID.')
def run(config, gpu_id=0):
    """Read the config file and run the experiment"""
    #-------check the format of config file:
    cfg = configuration.verify_config(config)
    # print(cfg)

    #-------setup mlflow
    configuration.setup_mlflow(cfg['mlflow'])

    #-------check cuda
    if cfg['training']['use_cuda']:
        use_cuda = torch.cuda.is_available()
        if use_cuda == False:
            print("[  use_cuda  ] No GPU available in your machine, will be using CPU")
        if use_cuda == True:
            device_name = torch.cuda.get_device_name(gpu_id)
            print(f"[  use_cuda  ]  will be using: {device_name}")
            torch.cuda.set_device(gpu_id)
    else:
        use_cuda = False

    #-------Getting training args
    epoch = cfg['training']["epoch"]
    batch_size = cfg['training']["batch_size"]
    
    #-------Setup datasets
    #if datasplit is true dataset that has only train, test; we will split the train to train and valid
    # if datasplit false, dataset has train, val, test itself.
    train_dataloader, val_dataloader, test_dataloader = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], batch_size)

    #-------Setup Head
    #TODO: mighe need it for MTL

    #-------setup model
    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    model = configuration.setup_model(cfg['model'])

    #-------start training
    train(train_dataloader, val_dataloader, test_dataloader, model, cfg['training'], use_cuda, cfg['optimizer'])


# def main():
#     #------------dataset details------------
#     DATA_DIR = "./data"
#     data_path = f'{DATA_DIR}/OpenI/OpenI_cheXpertLabels.csv'
#     split_path = f'{DATA_DIR}/OpenI/cheXpertLabels'
#     use_data_loader = True

#     epochs = 3 # Number of training epochs (authors recommend between 2 and 4)
#     batch_size = 16
#     max_length = 128
    
#     model_name = 'bert-base-uncased'
#     tokenizer_name = "bert-base-uncased"

#     # # model_name = "bert-base-cased"
#     # # tokenizer_name = "bert-base-cased"

#     ModelTokenizer = BertTokenizer
#     PreTrainedModel = BertCLS

#     #medical sense
#     group1 = [0,1,2,6]  #{'No Finding', 'Cardiomegaly','Lung Opacity','Atelectasis'}
#     group2 = [3,4,5,7,9] #{'Edema','Consolidation','Pneumonia', 'Pneumothorax','Fracture'}
#     group3 = [8,10] #{'Pleural Effusion','SupportDevices',}
#     heads_index = [group1, group2, group3]
#     col_names = ['No Finding', 'Cardiomegaly', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 
#                 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Fracture', 'SupportDevices']
#     multihead_cls(data_path,split_path, PreTrainedModel, epochs, batch_size, max_length ,
#                ModelTokenizer, tokenizer_name, model_name, use_data_loader, heads_index, col_names)


if __name__ == "__main__":
    main()