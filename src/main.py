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
import mtl.utils.logger as mlflowLogger 
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
    training_type = cfg['training']['type']
    
    #-------Setup datasets
    training_cv = cfg['training']['cv']
    fold = cfg['training']['fold']
    dataset_obj, dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], cfg['model'], batch_size, training_cv)
    if training_cv:
        results = []
        fold_i = 0
        for train_dataloader, val_dataloader, test_dataloader, num_labels in dataset_obj(dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg, training_cv, fold): 
            fold_i += 1
            print(f"[main] Fold {fold_i}")
            #-------setup dataset
            # train_dataloader, val_dataloader, test_dataloader, num_labels = dataset_obj(dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg, training_cv, fold)

            #-------setup model
            # Load model, the pretrained model will include a single linear classification layer on top for classification. 
            if training_type == "singlehead_cls":
                model = configuration.setup_model(cfg['model'])(num_labels, training_type)
            if training_type == "MTL_cls":
                heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index"))
                num_labels = [len(labels) for labels in heads_index]
                model = configuration.setup_model(cfg['model'])(num_labels, training_type, device)

            freeze = list(cfg['model'].values())[0]['freeze']
            if freeze:
                # model.freeze_bert_encoder()
                for param in model.base_model.parameters():
                    param.requires_grad = False

            #-------setup training
            test_f1_micro, test_f1_macro, test_acc = train(train_dataloader, val_dataloader, test_dataloader, model, cfg , use_cuda, fold_i)
            results.append([test_f1_micro, test_f1_macro, test_acc])
        #-------calculate mean and variance of run details
        results = np.array(results)
        mean = np.mean(results, axis=0)
        mlflowLogger.store_metric(f"test.f1_micro.mean", mean[0])       
        mlflowLogger.store_metric(f"test.f1_macro.mean", mean[1])       
        mlflowLogger.store_metric(f"test.acc.mean", mean[2])          
        
        std = np.std(results, axis=0)
        mlflowLogger.store_metric(f"test.f1_micro.std", std[0])       
        mlflowLogger.store_metric(f"test.f1_macro.std", std[1])       
        mlflowLogger.store_metric(f"test.acc.std", std[2])          
        
        mlflowLogger.finish_mlflowrun()
        
    else:
        train_dataloader, val_dataloader, test_dataloader, num_labels = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], cfg['model'], batch_size)

        #-------setup model
        # Load model, the pretrained model will include a single linear classification layer on top for classification. 
        if training_type == "singlehead_cls":
            model = configuration.setup_model(cfg['model'])(num_labels, training_type)
        if training_type == "MTL_cls":
            heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index"))
            num_labels = [len(labels) for labels in heads_index]
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