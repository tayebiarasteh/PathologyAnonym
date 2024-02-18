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
import timm
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.stats import ttest_ind

from config.serde import open_experiment, create_experiment, delete_experiment
from data.classification_data import Dataloader_disorder
from PathologyAnonym_Train_Valid import Training
from pathanonym_Prediction import Prediction

import warnings
warnings.filterwarnings('ignore')




def main_train_disorder_detection(global_config_path="/PATH/PathologyAnonym/config/config.yaml", valid=False,
                  resume=False, experiment_name='name'):
    """Main function for training + validation centrally

        Parameters
        ----------
        global_config_path: str
            always global_config_path="/PATH/PathologyAnonym/config/config.yaml"

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


    model = timm.create_model('resnet34', num_classes=2, pretrained=True)

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




def main_eval_test_disorder_detection(global_config_path="/PATH/PathologyAnonym/config/config.yaml",
                   experiment_name='name', avg_epochs=10, model_epoch=50):
    """Main function for testing.

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/PATH/PathologyAnonym/config/config.yaml"

    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    # model = timm.create_model('resnet18', num_classes=2, pretrained=True)
    model = timm.create_model('resnet34', num_classes=2, pretrained=True)
    # model = timm.create_model('resnet50', num_classes=2, pretrained=True)

    predictor = Prediction(cfg_path)
    predictor.setup_model(model=model, model_epoch=model_epoch)
    # average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision = 0,0,0,0,0,0
    average_f1_score = []
    average_AUROC = []
    average_accuracy = []
    average_specificity = []
    average_sensitivity = []
    average_precision = []

    for _ in tqdm(range(avg_epochs)):
        test_dataset = Dataloader_disorder(cfg_path=cfg_path, mode='test', experiment_name=experiment_name)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1,
                                                  pin_memory=True, drop_last=False, shuffle=False, num_workers=15)

        f1_score, AUROC, accuracy, specificity, sensitivity, precision = predictor.predict(test_loader=test_loader)

        average_f1_score.append(f1_score)
        average_AUROC.append(AUROC)
        average_accuracy.append(accuracy)
        average_specificity.append(specificity)
        average_sensitivity.append(sensitivity)
        average_precision.append(precision)

    final_f1_score = np.stack(average_f1_score).mean(1)
    final_AUROC = np.stack(average_AUROC).mean(1)
    final_accuracy = np.stack(average_accuracy).mean(1)
    final_specificity = np.stack(average_specificity).mean(1)
    final_sensitivity = np.stack(average_sensitivity).mean(1)
    final_precision = np.stack(average_precision).mean(1)

    predictor.savings_prints(valid_F1=final_f1_score, valid_AUC=final_AUROC, valid_accuracy=final_accuracy, valid_specificity=final_specificity,
                             valid_sensitivity=final_sensitivity, valid_precision=final_precision, avg_epochs=avg_epochs)




def pvalue_ttest(global_config_path="/PATH/PathologyAnonym/config/config.yaml", experiment_name='name',
                 df1_path="/PATH/stat_log/test_results.csv",
                   df2_path='/PATH/stat_log/test_results.csv'):

    params = open_experiment(experiment_name, global_config_path)

    # Read the CSV files into pandas DataFrames
    df_model1 = pd.read_csv(df1_path)
    df_model2 = pd.read_csv(df2_path)

    # Initialize a dictionary to hold the t-test results and additional statistics
    t_test_results = {}

    # Iterate over the columns to perform t-tests for each metric
    for column in df_model1.columns:
        # Perform an unpaired two-sided t-test for the current metric
        t_stat, p_value = ttest_ind(df_model1[column], df_model2[column], equal_var=False)

        # Calculate mean and standard deviation for each model
        mean_model1 = df_model1[column].mean()
        mean_model2 = df_model2[column].mean()
        std_model1 = df_model1[column].std()
        std_model2 = df_model2[column].std()

        # Store the results and additional statistics in the dictionary
        t_test_results[column] = {
            't-statistic': t_stat,
            'p-value': p_value,
            'mean_model1': mean_model1,
            'mean_model2': mean_model2,
            'std_model1': std_model1,
            'std_model2': std_model2
        }

    # Save the results to a text file
    results_file_path = 't_test_results.txt'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/pvalues', 'w') as file:
        for metric, results in t_test_results.items():
            file.write(f"{metric}:\n")
            file.write(f"t-statistic = {results['t-statistic']:.3f}, p-value = {results['p-value']:.3f}\n")
            file.write(
                f"Mean (Model 1) = {results['mean_model1'] * 100:.2f}%, Std Dev (Model 1) = {results['std_model1'] * 100:.2f}%\n")
            file.write(
                f"Mean (Model 2) = {results['mean_model2'] * 100:.2f}%, Std Dev (Model 2) = {results['std_model2'] * 100:.2f}%\n\n")

    print(f'Results saved to {results_file_path}')


