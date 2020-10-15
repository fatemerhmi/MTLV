import pandas as pd
import ast
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import numpy as np

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
            device = 'cpu'
        if use_cuda == True:
            device_name = torch.cuda.get_device_name(gpu_id)
            print(f"[  use_cuda  ]  will be using: {device_name}")
            device = 'cuda'
            torch.cuda.set_device(gpu_id)
    else:
        use_cuda = False

    #-------Getting training args
    epoch = cfg['training']["epoch"]
    batch_size = cfg['training']["batch_size"]
    
    #-------Setup datasets
    #if datasplit is true dataset that has only train, test; we will split the train to train and valid
    # if datasplit false, dataset has train, val, test itself.
    train_dataloader, val_dataloader, test_dataloader, num_labels = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], batch_size)

    #-------Setup Head
    #TODO: mighe need it for MTL

    #-------setup model
    # Load model, the pretrained model will include a single linear classification layer on top for classification. 
    training_type = cfg['training']['type']
    if training_type == "singlehead_cls":
        model = configuration.setup_model(cfg['model'])(num_labels, training_type)
    if training_type == "MTL_cls":
        num_labels = [len(labels) for labels in list(cfg['head'].values())[0]['heads_index']]
        model = configuration.setup_model(cfg['model'])(num_labels, training_type, device)

    freeze = list(cfg['model'].values())[0]['freeze']
    if freeze:
        # model.freeze_bert_encoder()
        for param in model.base_model.parameters():
            param.requires_grad = False

    #-------start training
    train(train_dataloader, val_dataloader, test_dataloader, model, cfg , use_cuda)

if __name__ == "__main__":
    main()