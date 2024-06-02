from recbole.config import Config
from recbole.data import create_dataset, get_dataloader, create_samplers, load_split_dataloaders
from recbole.utils import (
    init_seed,
    get_model,
    get_trainer
)
from static import *

def data_preparation(config, dataset):
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
        dataset._change_feat_format()
    else:
        built_datasets = dataset.build()
        train_dataset, valid_dataset, test_dataset = built_datasets
        train_sampler, valid_sampler, test_sampler = create_samplers(
            config, dataset, built_datasets
        )
        train_data = get_dataloader(config, "train")(
            config, train_dataset, train_sampler, shuffle=False
        )
        test_data = get_dataloader(config, "test")(
            config, test_dataset, test_sampler, shuffle=False
        )
    return train_data, test_data

def fit_recbole(params, file_name, recommender):
    config = Config(model=recommender, dataset=file_name, config_dict=params, config_file_list=[f"{recommender}.yaml"])
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_dataloader, test_dataloader = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_dataloader._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    return trainer, train_dataloader

def eval_recbole(params, file_name, recommender, model_name):
    config = Config(model=recommender, dataset=file_name, config_dict=params, config_file_list=[f"{recommender}.yaml"])
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_dataloader, test_dataloader = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_dataloader._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    trainer.resume_checkpoint(params["checkpoint_dir"] + model_name)
    test_result = trainer.evaluate(test_dataloader, load_best_model=True, show_progress=config["show_progress"])
    return test_result
