"""
PathologyAnonym_main.py
Created on Feb 5, 2024.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pdb
import os
from torch.utils.data import DataLoader
import torch
from torch import nn

from config.serde import open_experiment, create_experiment, delete_experiment
from data.classification_data import Dataloader_disorder
from PathologyAnonym_Train_Valid import Training
import timm

import warnings
warnings.filterwarnings('ignore')




def main_train_disorder_detection(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/config/config.yaml", valid=False,
                  resume=False, experiment_name='name'):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/config/config.yaml"

        valid: bool
            if we want to do validation

        resume: bool
            if we are resuming training on a model

        experiment_name: str
            name of the experiment, in case of resuming training.
            name of new experiment, in case of new training.
    """
    if resume == True:
        params = open_experiment(experiment_name, global_config_path)
    else:
        params = create_experiment(experiment_name, global_config_path)
    cfg_path = params["cfg_path"]

    train_dataset = Dataloader_disorder(cfg_path=cfg_path, mode='train', experiment_name=experiment_name)
    valid_dataset = Dataloader_disorder(cfg_path=cfg_path, mode='test', experiment_name=experiment_name)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1,
                                               pin_memory=True, drop_last=True, shuffle=True, num_workers=10)
    weight = train_dataset.pos_weight()
    # weight = None

    if valid:
        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=1,
                                                   pin_memory=True, drop_last=False, shuffle=False, num_workers=5)
    else:
        valid_loader = None


    # model = timm.create_model('resnet18', num_classes=2, pretrained=True)
    model = timm.create_model('resnet34', num_classes=2, pretrained=True)
    # model = timm.create_model('resnet50', num_classes=2, pretrained=True)

    loss_function = nn.BCEWithLogitsLoss

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params['Network']['lr']),
                                 weight_decay=float(params['Network']['weight_decay']),
                                 amsgrad=params['Network']['amsgrad'])

    trainer = Training(cfg_path, resume=resume)
    if resume == True:
        trainer.load_checkpoint(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    else:
        trainer.setup_model(model=model, optimiser=optimizer, loss_function=loss_function, weight=weight)
    trainer.train_epoch(train_loader=train_loader, valid_loader=valid_loader, num_epochs=params['num_epochs'])







if __name__ == '__main__':
    cfg_path = "/home/soroosh/Documents/Repositories/PathologyAnonym/config/config.yaml"
    # delete_experiment(experiment_name='dysarthria_70_30_contentmel_anonym', global_config_path=cfg_path)

    main_train_disorder_detection(global_config_path=cfg_path, valid=True, resume=False, experiment_name='dysarthria_70_30_contentmel')
    # main_train_disorder_detection(global_config_path=cfg_path, valid=True, resume=False, experiment_name='dysarthria_70_30_contentmel_anonym')
