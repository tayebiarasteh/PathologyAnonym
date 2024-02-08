"""
Created on Feb 7, 2024.
pathanonym_Prediction.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import time
import random
import pdb
from tqdm import tqdm
import os
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
from sklearn import metrics

from config.serde import read_config
epsilon = 1e-15


class Prediction:
    def __init__(self, cfg_path):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.setup_cuda()


    def setup_cuda(self, cuda_device_id=0):
        """setup the device.

        Parameters
        ----------
        cuda_device_id: int
            cuda device id
        """
        if torch.cuda.is_available():
            torch.backends.cudnn.fastest = True
            torch.cuda.set_device(cuda_device_id)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')


    def time_duration(self, start_time, end_time):
        """calculating the duration of training or one iteration

        Parameters
        ----------
        start_time: float
            starting time of the operation

        end_time: float
            ending time of the operation

        Returns
        -------
        elapsed_hours: int
            total hours part of the elapsed time

        elapsed_mins: int
            total minutes part of the elapsed time

        elapsed_secs: int
            total seconds part of the elapsed time
        """
        elapsed_time = end_time - start_time
        elapsed_hours = int(elapsed_time / 3600)
        if elapsed_hours >= 1:
            elapsed_mins = int((elapsed_time / 60) - (elapsed_hours * 60))
            elapsed_secs = int(elapsed_time - (elapsed_hours * 3600) - (elapsed_mins * 60))
        else:
            elapsed_mins = int(elapsed_time / 60)
            elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_hours, elapsed_mins, elapsed_secs


    def setup_model(self, model, model_file_name=None, model_epoch=400):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']
        self.model = model.to(self.device)
        # self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name)))
        self.model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], "epoch" + str(model_epoch) +"_" + model_file_name)))



    def savings_prints(self, valid_F1=None, valid_AUC=None, valid_accuracy=None,
                       valid_specificity=None, valid_sensitivity=None, valid_precision=None, avg_epochs=10):
        """Saving the model weights, checkpoint, information,
        and training and validation loss and evaluation statistics.

        """
        print('------------------------------------------------------'
              '----------------------------------')

        print(f'\t Results over {str(avg_epochs)} repetition of the testing.\n\n AUROC: {valid_AUC.mean() * 100:.2f} ± {valid_AUC.std() * 100:.2f}% | accuracy: {valid_accuracy.mean() * 100:.2f} ± {valid_accuracy.std() * 100:.2f}%'
        f' | F1: {valid_F1.mean() * 100:.2f} ± {valid_F1.std() * 100:.2f}% | specificity: {valid_specificity.mean() * 100:.2f} ± {valid_specificity.std() * 100:.2f}%'
        f' | recall (sensitivity): {valid_sensitivity.mean() * 100:.2f} ± {valid_sensitivity.std() * 100:.2f}%\n')

        # saving the training and validation stats
        msg = f'\n\n----------------------------------------------------------------------------------------\n' \
               f'Results over {str(avg_epochs)} repetition of the testing.\n\n AUROC: {valid_AUC.mean() * 100:.2f} ± {valid_AUC.std() * 100:.2f}% | accuracy: {valid_accuracy.mean() * 100:.2f} ± {valid_accuracy.std() * 100:.2f}% ' \
               f' | F1: {valid_F1.mean() * 100:.2f} ± {valid_F1.std() * 100:.2f}% | specificity: {valid_specificity.mean() * 100:.2f} ± {valid_specificity.std() * 100:.2f}%' \
               f' | recall (sensitivity): {valid_sensitivity.mean() * 100:.2f} ± {valid_sensitivity.std() * 100:.2f}%\n\n'
        with open(os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_results', 'a') as f:
            f.write(msg)
        
        
        # Create a dictionary with your data, where keys are column names and values are the arrays.
        data = {
            'AUC': valid_AUC,
            'accuracy': valid_accuracy,
            'f1_score': valid_F1,
            'specificity': valid_specificity,
            'sensitivity': valid_sensitivity,
            'precision': valid_precision
        }

        # Convert the dictionary into a pandas DataFrame.
        df = pd.DataFrame(data)

        # Save the DataFrame to a CSV file.
        csv_file_path = os.path.join(self.params['target_dir'], self.params['stat_log_path']) + '/test_results.csv'  # Specify your desired path and file name.
        df.to_csv(csv_file_path, index=False)  # `index=False` to not include row indices in the CSV.





    def predict(self, test_loader):
        """
        Returns
        -------
        """
        self.model.eval()
        total_f1_score = []
        total_AUROC = []
        total_accuracy = []
        total_specificity_score = []
        total_sensitivity_score = []
        total_precision_score = []

        # initializing the caches
        preds_with_sigmoid_cache = torch.Tensor([]).to(self.device)
        logits_for_loss_cache = torch.Tensor([]).to(self.device)
        labels_cache = torch.Tensor([]).to(self.device)

        for idx, (image, label) in enumerate(test_loader):

            image = image.squeeze(0)
            label = label.squeeze(0)
            image = image.to(self.device)
            label = label.to(self.device)
            image = image.float()

            with torch.no_grad():
                output = self.model(image)

                output_sigmoided = F.sigmoid(output)

                # saving the logits and labels of this batch
                preds_with_sigmoid_cache = torch.cat((preds_with_sigmoid_cache, output_sigmoided))
                logits_for_loss_cache = torch.cat((logits_for_loss_cache, output))
                labels_cache = torch.cat((labels_cache, label))

        ############ Evaluation metric calculation ########

        # threshold finding for metrics calculation
        preds_with_sigmoid_cache = preds_with_sigmoid_cache.cpu().numpy()
        labels_cache = labels_cache.int().cpu().numpy()
        optimal_threshold = np.zeros(labels_cache.shape[1])

        for idx in range(labels_cache.shape[1]):
            fpr, tpr, thresholds = metrics.roc_curve(labels_cache[:, idx], preds_with_sigmoid_cache[:, idx], pos_label=1)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold[idx] = thresholds[optimal_idx]

        predicted_labels = (preds_with_sigmoid_cache > optimal_threshold).astype(np.int32)

        confusion = metrics.multilabel_confusion_matrix(labels_cache, predicted_labels)

        F1_disease = []
        accuracy_disease = []
        specificity_disease = []
        sensitivity_disease = []
        precision_disease = []

        for idx, disease in enumerate(confusion):
            TN = disease[0, 0]
            FP = disease[0, 1]
            FN = disease[1, 0]
            TP = disease[1, 1]
            F1_disease.append(2 * TP / (2 * TP + FN + FP + epsilon))
            accuracy_disease.append((TP + TN) / (TP + TN + FP + FN + epsilon))
            specificity_disease.append(TN / (TN + FP + epsilon))
            sensitivity_disease.append(TP / (TP + FN + epsilon))
            precision_disease.append(TP / (TP + FP + epsilon))

        # Macro averaging
        total_f1_score.append(np.stack(F1_disease))
        try:
            total_AUROC.append(metrics.roc_auc_score(labels_cache, preds_with_sigmoid_cache, average=None))
        except:
            print('hi')
            pass
        total_accuracy.append(np.stack(accuracy_disease))
        total_specificity_score.append(np.stack(specificity_disease))
        total_sensitivity_score.append(np.stack(sensitivity_disease))
        total_precision_score.append(np.stack(precision_disease))

        average_f1_score = np.stack(total_f1_score).mean(0)
        average_AUROC = np.stack(total_AUROC).mean(0)
        average_accuracy = np.stack(total_accuracy).mean(0)
        average_specificity = np.stack(total_specificity_score).mean(0)
        average_sensitivity = np.stack(total_sensitivity_score).mean(0)
        average_precision = np.stack(total_precision_score).mean(0)

        return average_f1_score, average_AUROC, average_accuracy, average_specificity, average_sensitivity, average_precision
