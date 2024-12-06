import json
from src.model.vocab import Vocab

class TrainConst:
    r""" Training config here """

    with open("./src/config/default_training_config.json", 'r') as f:
        config = json.load(f)
    # General
    random_seed = 42
    is_pretrained = config["model"]["cnn"]["pretrained"]
    device = config["model"]["device"]
    pretrained_weight = config["model"]["pretrain"]

    # Training
    batch_size = config["trainer"]["batch_size"]
    print_every = config["trainer"]["print_every"]
    valid_every = config["trainer"]["valid_every"]
    iters = config["trainer"]["iters"]
    checkpoint = config["trainer"]["checkpoint"]
    export = config["trainer"]["export"]
    metrics = config["trainer"]["metrics"]
        
        
class DataConst:
    r""" Dataset config here"""

    with open("./src/config/default_training_config.json", 'r') as f:
        config = json.load(f)

    vocab = Vocab(config["model"]["vocab"])
    name = config["dataset"]["name"]
    data_root = config["dataset"]["data_root"]
    train_annotation = config["dataset"]["train_annotation"]
    valid_annotation = config["dataset"]["valid_annotation"]
    image_height = config['dataset']['image_height']
    image_min_width=config['dataset']['image_min_width']
    image_max_width=config['dataset']['image_max_width']
    n_sample = config['dataset']['n_sample']
    batch_size = config["trainer"]["batch_size"]
    bucket_image_name = config["dataset"]["bucket_image_name"]
    dataloader = config["dataloader"]

class EnvConst:
    r""" Envir config here"""

    with open("./src/config/env_config.json", 'r') as f:
        config = json.load(f)
    
     # Assigning values from the loaded config
    mysql_username = config["mysql"]["username"]
    mysql_host = config["mysql"]["host"]
    mysql_port = config["mysql"]["port"]
    mysql_password = config["mysql"]["password"]
    mysql_database = config["mysql"]["database"]

    mongodb_username = config["mongodb"]["username"]
    mongodb_password = config["mongodb"]["password"]
    mongodb_host = config["mongodb"]["host"]
    mongodb_port = config["mongodb"]["port"]
    mongodb_database_name = config["mongodb"]["database_name"]
    mongodb_collection_name = config["mongodb"]["collection_name"]

    minio_url = config["minio"]["url"]
    minio_access_key = config["minio"]["access_key"]
    minio_secret_key = config["minio"]["secret_key"]
    minio_bucket_name = "ocrannotationrotate"
    
    
    
class PredictConst:
    pass
        
        
        
        
    
    

