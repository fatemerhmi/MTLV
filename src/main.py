import pandas as pd
import ast
import torch
import numpy as np
from scipy import stats
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

def store_mflow_metrics(results, prefix):
    mean = np.mean(results, axis=0) 
    mlflowLogger.store_metric(f"{prefix}.cv.test.f1_micro.mean", mean[0])       
    mlflowLogger.store_metric(f"{prefix}.cv.test.f1_macro.mean", mean[1])            
    mlflowLogger.store_metric(f"{prefix}.cv.test.hamming_score.mean", mean[2])          
    mlflowLogger.store_metric(f"{prefix}.cv.test.subset_accuracy.mean", mean[3])          
    
    std = np.std(results, axis=0)
    mlflowLogger.store_metric(f"{prefix}.cv.test.f1_micro.std", std[0])       
    mlflowLogger.store_metric(f"{prefix}.cv.test.f1_macro.std", std[1])                
    mlflowLogger.store_metric(f"{prefix}.cv.test.hamming_score.std", std[2])          
    mlflowLogger.store_metric(f"{prefix}.cv.test.subset_accuracy.std", std[3])   

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
        elif use_cuda == True:
            device_name = torch.cuda.get_device_name(gpu_id)
            print(f"[  use_cuda  ]  will be using: {device_name}")
            device = 'cuda'
            torch.cuda.set_device(gpu_id)
    else:
        use_cuda = False

    #-------Getting training args
    training_type = cfg['training']['type']
    if training_type!="ttest":
        epoch = cfg['training']["epoch"]
    batch_size = cfg['training']["batch_size"]
    
    #-------Setup datasets
    if "cv" in cfg['training'].keys():
        training_cv = cfg['training']['cv']
        if training_cv == False:
            NameError("Fold is given, but cv is False. Make sure to put cv as True if you want to do cross validation")
        fold = cfg['training']['fold']
        dataset_obj, dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], cfg['model'], batch_size, training_cv)
    else:
        training_cv = False

    #================================ttest==================================
    if training_type == "ttest":
        training_cv = True
        fold = 10
        dataset_obj, dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], cfg['model'], batch_size, training_cv, ttest = True)
        results_MTL = []
        results_singleHead = []
        fold_i = 0

        for train_dataloader_s, val_dataloader_s, test_dataloader_s, num_labels_s, train_dataloader_mtl, val_dataloader_mtl, test_dataloader_mtl, num_labels_mtl in dataset_obj(dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold): 
            fold_i += 1
            print(f"[main] Fold {fold_i}")

            #-------single head--------
            model_singleHead = configuration.setup_model(cfg['model'])(num_labels_s, "singlehead_cls")
            test_f1_micro, test_f1_macro, test_subset_accuracy, test_hamming_score_ = train(train_dataloader_s, val_dataloader_s, test_dataloader_s, model_singleHead, cfg , use_cuda, "singlehead_cls", fold_i)
            results_singleHead.append([test_f1_micro, test_f1_macro, test_hamming_score_, test_subset_accuracy])
            
            #-------MTL----------------
            heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index"))
            num_labels_head = [len(labels) for labels in heads_index]
            model_MTL = configuration.setup_model(cfg['model'])(num_labels_head, "MTL_cls", device)
            test_f1_score_micro, test_f1_score_macro, test_hamming_score_, test_subset_accuracy = train(train_dataloader_mtl, val_dataloader_mtl, test_dataloader_mtl, model_MTL, cfg , use_cuda, "MTL_cls", fold_i)
            results_MTL.append([test_f1_score_micro, test_f1_score_macro, test_hamming_score_, test_subset_accuracy])


        #-------calculate mean and variance of run details------
        # mlflowLogger.store_metric(f"results_MTL", results_MTL)
        # mlflowLogger.store_metric(f"results_singleHead", results_singleHead)

        # ttest
        results_MTL = np.array(results_MTL)
        store_mflow_metrics(results_MTL, "MTL")
        results_singleHead = np.array(results_singleHead)
        store_mflow_metrics(results_singleHead, "STL")

        _, ttest_f1_mi =  stats.ttest_rel(results_MTL[:,0],results_singleHead[:,0])
        _, ttest_f1_ma =  stats.ttest_rel(results_MTL[:,1],results_singleHead[:,1])
        _, ttest_hammingscore   =  stats.ttest_rel(results_MTL[:,2],results_singleHead[:,2])
        _, ttest_subset_acc   =  stats.ttest_rel(results_MTL[:,3],results_singleHead[:,3])

        mlflowLogger.store_metric(f"ttest_f1_mi", ttest_f1_mi)          
        mlflowLogger.store_metric(f"ttest_f1_ma", ttest_f1_ma)          
        mlflowLogger.store_metric(f"ttest_hammingscore", ttest_hammingscore)          
        mlflowLogger.store_metric(f"ttest_subset_acc", ttest_subset_acc)          

        # ttest wilcoxon
        _, ttest_f1_mi =  stats.wilcoxon(results_MTL[:,0],results_singleHead[:,0])
        _, ttest_f1_ma =  stats.wilcoxon(results_MTL[:,1],results_singleHead[:,1])
        _, ttest_hammingscore   =  stats.wilcoxon(results_MTL[:,2],results_singleHead[:,2])
        _, ttest_subset_acc   =  stats.wilcoxon(results_MTL[:,3],results_singleHead[:,3])

        mlflowLogger.store_metric(f"ttest_f1_wilcoxon_mi", ttest_f1_mi)          
        mlflowLogger.store_metric(f"ttest_f1_wilcoxon_ma", ttest_f1_ma)          
        mlflowLogger.store_metric(f"ttest_wilcoxon_hammingscore", ttest_hammingscore)               
        mlflowLogger.store_metric(f"ttest_wilcoxon_subset_acc", ttest_subset_acc)               

        mlflowLogger.finish_mlflowrun()
        return

    #==================================cross validation========================================
    elif training_cv: #cross validation
        results = []
        fold_i = 0
        for train_dataloader, val_dataloader, test_dataloader, num_labels in dataset_obj(dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size, model_cfg, fold): 
            fold_i += 1
            print(f"[main] Fold {fold_i}")

            #-------setup model
            # Load model, the pretrained model will include a single linear classification layer on top for classification. 
            if training_type == "singlehead_cls":
                model = configuration.setup_model(cfg['model'])(num_labels, training_type)
            elif training_type == "MTL_cls":
                heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index"))
                num_labels = [len(labels) for labels in heads_index]
                model = configuration.setup_model(cfg['model'])(num_labels, training_type, device)

            freeze = list(cfg['model'].values())[0]['freeze']
            if freeze:
                # model.freeze_bert_encoder()
                for param in model.base_model.parameters():
                    param.requires_grad = False

            #-------setup training
            test_f1_score_micro, test_f1_score_macro, test_hamming_score_, test_subset_accuracy = train(train_dataloader, val_dataloader, test_dataloader, model, cfg , use_cuda, training_type, fold_i)
            results.append([test_f1_score_micro, test_f1_score_macro, test_hamming_score_, test_subset_accuracy])
        #-------calculate mean and variance of run details
        results = np.array(results)
        if training_type == "singlehead_cls":
            store_mflow_metrics(results, "STL")
        elif training_type == "MTL_cls":
            store_mflow_metrics(results, "MTL")
         
        
        mlflowLogger.finish_mlflowrun()
        return

    #==================================single run/ NO cross validation========================================
    else:
        train_dataloader, val_dataloader, test_dataloader, num_labels = configuration.setup_dataset(cfg['dataset'], cfg['tokenizer'], cfg['head'], cfg['model'], batch_size)

        #-------setup model
        # Load model, the pretrained model will include a single linear classification layer on top for classification. 
        if training_type == "singlehead_cls":
            model = configuration.setup_model(cfg['model'])(num_labels, training_type)
        elif training_type == "MTL_cls":
            heads_index = ast.literal_eval(mlflowLogger.get_params("heads_index"))
            num_labels = [len(labels) for labels in heads_index]
            model = configuration.setup_model(cfg['model'])(num_labels, training_type, device)

        freeze = list(cfg['model'].values())[0]['freeze']
        if freeze:
            # model.freeze_bert_encoder()
            for param in model.base_model.parameters():
                param.requires_grad = False

        #-------start training
        train(train_dataloader, val_dataloader, test_dataloader, model, cfg , use_cuda, training_type)
        return
        
if __name__ == "__main__":
    main()