import torch
import yaml

__all__ = ["setup_model", "setup_datsaet", "load_config"]

def setup_model():
    pass

def setup_datsaet():
    pass

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

    required_keys = ['dataset_name', 'model', 'training', 'head', 'tokenizer']
    if any(key not in cfg for key in required_keys):
        raise Exception(f"Missing key in configuration file. Required keys are: {required_keys}")
    
    # training:
    training_keys = ['epoch', 'batch_size']
    if any(key not in cfg['training'] for key in training_keys):
        raise Exception(f"Missing key in training part. Required keys are: {required_keys}")

    # Head:
    if cfg['head']['count']<1:
        raise Exception("Number of heads shouls at least be one.")
