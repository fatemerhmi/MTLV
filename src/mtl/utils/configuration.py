import torch
import yaml
import importlib
import ast

# from mtl.datasets import *
import mtl.datasets as datasets
import mtl.tokenizers as tokenizers
import mtl.models as models
import mtl.optimizers as optimizers
import mtl.utils.logger as mlflowLogger 
import mtl.utils.loss as losses

__all__ = ["setup_model", "setup_dataset", "load_config", "setup_optimizer", "setup_mlflow"]

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
        raise NameError(f'{model_args["model_name"]} does not appear in this lists of models we support: {available_model_type_list}')
    
    #-------log model name and args to mlflow
    mlflowLogger.store_param("model", model_name)
    for arg in model_args:
        mlflowLogger.store_param("model."+model_family+"."+arg , model_args[arg])

    return model_obj

def setup_dataset(dataset_cfg, tokenizer_cfg, head_cfg, batch_size):

    #-------dataset name and args:
    dataset_name = list(dataset_cfg.keys())[0]
    dataset_args = list(dataset_cfg.values())[0]
    print(f"[  setup dataset  ] datset name: {dataset_name}, dataset arge: {dataset_args}")

    #-------log dataset name and args to mlflow
    mlflowLogger.store_param("dataset", dataset_name)
    for arg in dataset_args:
        mlflowLogger.store_param("dataset."+dataset_name+"."+arg , dataset_args[arg])

    #-------tokenizer name and args:
    tokenizer_name = list(tokenizer_cfg.keys())[0]
    tokenizer_args = list(tokenizer_cfg.values())[0]
    print(f"[  setup tokenizer  ] tokenizer name: {tokenizer_name}, Tokenizer arge: {tokenizer_args}")

    #-------log tokenizer name and args to mlflow
    mlflowLogger.store_param("tokenizer", tokenizer_name)
    for arg in tokenizer_args:
        mlflowLogger.store_param("tokenizer."+tokenizer_name+"."+arg , tokenizer_args[arg])

    #-------head name and args
    head_type = list(head_cfg.keys())[0]
    head_args = list(head_cfg.values())[0]
    print(f"[  setup head  ] head type: {head_type}, head arge: {head_args}")

    #-------log head name and args to mlflow
    mlflowLogger.store_param("head", head_type)
    for arg in head_args:
        mlflowLogger.store_param("head."+head_type+"."+arg , head_args[arg])

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

def setup_optimizer(optimizer_cfg, model_parameters):
    #-------optimizer name and args:
    optimizer_type = list(optimizer_cfg.keys())[0]
    optimizer_args = list(optimizer_cfg.values())[0]
    print(f"[  setup optimizer  ] optimizer type: {optimizer_type}, optimizer arge: {optimizer_args}")
    available_optimizers = optimizers.__all__
    if optimizer_type in available_optimizers:
        imported_optimizer_module = importlib.import_module( "mtl.optimizers."+ optimizer_type)
    else:
        raise NameError(f'{optimizer_type} does not appear in this lists of optimizers we support: {available_optimizers}')

    #-------get the optimizer obj
    optimizer_name = optimizer_args['name']
    if optimizer_name in imported_optimizer_module.__all__:
        optimizer_obj = getattr(imported_optimizer_module,optimizer_name)
    else:
        raise NameError(f'{optimizer_name} does not appear in this lists of optimizers we support: {available_optimizers}')
    
    #-------log optimizer name and args to mlflow
    mlflowLogger.store_param("optimizer", optimizer_type)
    for arg in optimizer_args:
        mlflowLogger.store_param("optimizer."+optimizer_type+"."+arg , optimizer_args[arg])
    # print(optimizer_obj)
    return optimizer_obj(params = model_parameters, lr = optimizer_args['lr'])

def setup_loss(loss_cfg):
    #-------loss type and args:
    loss_type = loss_cfg['type']
    loss_args = loss_cfg
    print(f"[  setup loss  ] loss type: {loss_type}, loss arge: {loss_args}")
    available_loss = losses.__all__
    imported_loss_modules = importlib.import_module( "mtl.utils.loss")

    #-------get the loss func
    if loss_type in imported_loss_modules.__all__:
        loss_func = getattr(imported_loss_modules,loss_type)
    else:
        raise NameError(f'{loss_type} does not appear in this lists of loss functions we support: {available_loss}')
    
    #-------log loss type and args to mlflow
    for arg in loss_args:
        mlflowLogger.store_param("loss."+arg , loss_args[arg])

    return loss_func

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

    required_keys = ['dataset', 'model', 'training', 'head', 'tokenizer', 'loss', 'optimizer']
    if any(key not in cfg for key in required_keys):
        raise Exception(f"Missing key in configuration file. Required keys are: {required_keys}")
    
    # training:
    training_keys = ['epoch', 'batch_size']
    if any(key not in cfg['training'] for key in training_keys):
        raise Exception(f"Missing key in training part. Required keys are: {required_keys}")

    # Head:
    head_args = list(cfg['head'].values())[0]
    if head_args['count']<1:
        raise Exception("Number of heads should at least be one.")

    # the len of head_index should be the same as count
    # print("____")
    head_type = list(cfg['head'].keys())[0]
    if head_type == "multi-head":
        if int(head_args['count']) != len(head_args['heads_index']):
            raise Exception("In multi-head configuration, the count and length of head_index should match!")

    # if loss config is weightedloss, then its len should match with the head count 
    loss_args = cfg['loss']
    if loss_args['type'] == "weightedloss":
        if len(loss_args['weights']) != int(head_args['count']):
            raise Exception("In multi-head configuration, the head count and length of weights should match!")
    
    # check if the sum of all weights will be 1:
    if loss_args['type'] == "weightedloss":
        if sum(ast.literal_eval(loss_args['weights'])) != 1:
            raise Exception("The sum of loss weights should be 1!")

    return cfg

def setup_mlflow(mlflow_cfg):
    experiment_name = mlflow_cfg['experiment_name']
    run_name = mlflow_cfg['run_name']
    tracking_uri = mlflow_cfg['tracking_uri']
    mlflowLogger.setup_mlflow(experiment_name, tracking_uri, run_name)
    print("[  setup mlflow ] mlflow successfuly set up!")