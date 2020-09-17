import torch
import yaml
import importlib

# from mtl.datasets import *
import mtl.datasets as datasets
import mtl.tokenizers as tokenizers

__all__ = ["setup_model", "setup_datsaet", "load_config"]

def setup_model():
    pass


def setup_dataset(dataset_cfg, tokenizer_cfg):

    #-------dataset name and args:
    dataset_name = list(dataset_cfg.keys())[0]
    dataset_args = list(dataset_cfg.values())[0]
    print(f"[  setup dataset ] datset name: {dataset_name}, dataset arge: {dataset_args}")

    #-------tokenizer name and args:
    tokenizer_name = list(tokenizer_cfg.keys())[0]
    tokenizer_args = list(tokenizer_cfg.values())[0]
    print(f"[  setup tokenizer ] tokenizer name: {tokenizer_name}, Tokenizer arge: {tokenizer_args}")

    #-------get all the available tokenizers:
    available_tokenizers = tokenizers.__all__
    if tokenizer_name in available_tokenizers:
        imported_tokenizer_module = importlib.import_module( "mtl.tokenizers."+ tokenizer_name)
    else:
        raise NameError(f'{tokenizer_name} does not appear in this lists of datasets we support: {available_tokenizers}')
    
    #-------get the tokenizer obj
    if tokenizer_args['name'] in imported_tokenizer_module.__all__:
        tokenizer_obj = getattr(imported_tokenizer_module,tokenizer_args['name'])()
        print(tokenizer_obj)
    else:
        raise NameError(f'{tokenizer_args["name"]} does not appear in this lists of tokenizers we support: {imported_tokenizer_module.__all__}')

    #-------get dataset object:
    available_datsets = datasets.__all__
    if dataset_name in available_datsets:
        imported_dataset_module = importlib.import_module( "mtl.datasets."+ dataset_name)
    else:
        raise NameError(f'{dataset_name} does not appear in this lists of datasets we support: {available_datsets}')

    train_dataset, valid_dataset, test_dataset = getattr(imported_dataset_module, "_setup_datasets")(dataset_name, tokenizer_obj, tokenizer_args)

    return train_dataset, valid_dataset, test_dataset

def load_config(config_file):

    default_config = {
        'experiment_name': "MTL_experiment",
    }

    with open(config_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    default_config.update(yaml_cfg)
    return default_config

def verify_config(config):
    cfg = load_config(config)

    required_keys = ['dataset', 'model', 'training', 'head', 'tokenizer']
    if any(key not in cfg for key in required_keys):
        raise Exception(f"Missing key in configuration file. Required keys are: {required_keys}")
    
    # training:
    training_keys = ['epoch', 'batch_size']
    if any(key not in cfg['training'] for key in training_keys):
        raise Exception(f"Missing key in training part. Required keys are: {required_keys}")

    # Head:
    if cfg['head']['count']<1:
        raise Exception("Number of heads shouls at least be one.")

    return cfg
