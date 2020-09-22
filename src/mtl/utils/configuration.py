import torch
import yaml
import importlib

# from mtl.datasets import *
import mtl.datasets as datasets
import mtl.tokenizers as tokenizers
import mtl.models as models

__all__ = ["setup_model", "setup_dataset", "load_config", "setup_optimizer"]

def setup_model(model_cfg):
    #-------model name and args:
    model_family = list(model_cfg.keys())[0]
    model_args = list(model_cfg.values())[0]
    print(f"[  setup model  ] model family name: {model_family}, model arge: {model_args}")
    available_models = models.__all__
    if model_family in available_models:
        imported_model_module = importlib.import_module( "mtl.models."+ model_family)
    else:
        raise NameError(f'{model_family} does not appear in this lists of models we support: {available_models}')
    
    #-------get the model obj
    model_name = model_args['model_name'].replace("-", "_")
    if model_name in imported_model_module.__all__:
        model_obj = getattr(imported_model_module,model_name)()
        # print(tokenizer_obj)
    else:
        available_model_type_list = str(imported_model_module.__all__).replace("_", "-")
        raise NameError(f'{model_args["model_name"]} does not appear in this lists of tokenizers we support: {available_model_type_list}')
    
    return model_obj

def setup_dataset(dataset_cfg, tokenizer_cfg, head_cfg, batch_size):

    #-------dataset name and args:
    dataset_name = list(dataset_cfg.keys())[0]
    dataset_args = list(dataset_cfg.values())[0]
    print(f"[  setup dataset  ] datset name: {dataset_name}, dataset arge: {dataset_args}")

    #-------tokenizer name and args:
    tokenizer_name = list(tokenizer_cfg.keys())[0]
    tokenizer_args = list(tokenizer_cfg.values())[0]
    print(f"[  setup tokenizer  ] tokenizer name: {tokenizer_name}, Tokenizer arge: {tokenizer_args}")

    #-------head name and args
    head_type = list(head_cfg.keys())[0]
    head_args = list(head_cfg.values())[0]
    print(f"[  setup head  ] head type: {head_type}, head arge: {head_args}")

    #-------get all the available tokenizers:
    available_tokenizers = tokenizers.__all__
    if tokenizer_name in available_tokenizers:
        imported_tokenizer_module = importlib.import_module( "mtl.tokenizers."+ tokenizer_name)
    else:
        raise NameError(f'{tokenizer_name} does not appear in this lists of datasets we support: {available_tokenizers}')
    
    #-------get the tokenizer obj
    if tokenizer_args['name'] in imported_tokenizer_module.__all__:
        tokenizer_obj = getattr(imported_tokenizer_module,tokenizer_args['name'])()
        # print(tokenizer_obj)
    else:
        raise NameError(f'{tokenizer_args["name"]} does not appear in this lists of tokenizers we support: {imported_tokenizer_module.__all__}')

    #-------get dataset object:
    available_datsets = datasets.__all__
    if dataset_name in available_datsets:
        imported_dataset_module = importlib.import_module( "mtl.datasets."+ dataset_name)
    else:
        raise NameError(f'{dataset_name} does not appear in this lists of datasets we support: {available_datsets}')

    train_dataloader, val_dataloader, test_dataloader = getattr(imported_dataset_module, "_setup_datasets")(dataset_name, dataset_args, tokenizer_obj, tokenizer_args, head_type, head_args, batch_size)

    return train_dataloader, val_dataloader, test_dataloader

def setup_optimizer(optimizer_cfg):


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
    head_args = list(cfg['head'].values())[0]
    if head_args['count']<1:
        raise Exception("Number of heads shouls at least be one.")

    return cfg
