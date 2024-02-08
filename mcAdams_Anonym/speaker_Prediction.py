"""
speaker_prediction.py
Created on Nov 22, 2021.
Prediction (test) class = evaluation + testing

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import time
import random
import pdb
from tqdm import tqdm
import os
import numpy as np
from matplotlib import pyplot as plt
import torch
import pandas as pd
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from inference_speaker_data_loader import original_dvector_loader, anonymized_dvector_loader
from config.serde import read_config


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



    def setup_model_for_inference(self, model, model_file_name=None):
        if model_file_name == None:
            model_file_name = self.params['trained_model_name']

        model.load_state_dict(torch.load(os.path.join(self.params['target_dir'], self.params['network_output_path'], model_file_name), map_location=torch.device('cpu')))
        self.model = model.to(self.device)


    def dvector_prediction_foranonym(self, test_loader):
        """
        Prediction
        For d-vector creation (prediction of the input utterances)

        for Autovc approach

        """
        self.params = read_config(self.cfg_path)
        self.model.eval()

        with torch.no_grad():
            # loop over speakers
            for speaker_name in tqdm(test_loader):
                embeddings_list = []
                speaker = test_loader[speaker_name]
                # loop over utterances
                for utterance in speaker:

                    features = []
                    # sliding window
                    for i in range(utterance.shape[0] // 80):
                        if i == (utterance.shape[0] // 80) - 1:
                            features.append(utterance[-160:])
                        else:
                            features.append(utterance[i * 80: i * 80 + 160])
                    try:
                        features = torch.stack(features)
                    except:
                        continue
                    features = features.to(self.device)

                    dvector = self.model(features)
                    dvector = torch.mean(dvector, dim=0)
                    dvector = dvector.cpu().numpy()
                    embeddings_list.append(dvector)

                embeddings = np.array(embeddings_list)
                # save embedding as numpy file
                np.save(os.path.join(os.path.join(self.params['target_dir'], self.params['dvectors_foranonym_path']),
                                     str(speaker_name) + ".npy"), embeddings.mean(0))


    # below needed
    def dvector_prediction(self, test_loader, anonymized=False):
        """
        Prediction
        For d-vector creation (prediction of the input utterances)
        """
        self.params = read_config(self.cfg_path)
        self.model.eval()

        with torch.no_grad():
            # loop over speakers
            for speaker_name in tqdm(test_loader):
                embeddings_list = []
                speaker = test_loader[speaker_name]
                if len(speaker) < 5:
                    continue
                # loop over utterances
                for utterance in speaker:

                    features = []
                    # sliding window
                    for i in range(utterance.shape[0]//80):
                        if i == (utterance.shape[0]//80) - 1:
                            features.append(utterance[-160:])
                        else:
                            features.append(utterance[i * 80: i * 80 + 160])
                    try:
                        features = torch.stack(features)
                    except:
                        continue
                    features = features.to(self.device)

                    dvector = self.model(features)
                    dvector = torch.mean(dvector, dim=0)
                    dvector = dvector.cpu().numpy()
                    embeddings_list.append(dvector)

                embeddings = np.array(embeddings_list)
                # save embedding as numpy file
                if anonymized:
                    np.save(os.path.join(os.path.join(self.params['target_dir'], self.params['dvectors_path_anonymized']), str(speaker_name) + ".npy"), embeddings)
                else:
                    np.save(os.path.join(os.path.join(self.params['target_dir'], self.params['dvectors_path_original']), str(speaker_name) + ".npy"), embeddings)


    # below needed
    def EER_newmethod_epochy(self, cfg_path, M=8, epochs=100):
        """
        evaluation (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        final_eer = []

        for _ in tqdm(range(epochs)):
            dvector_dataset = original_dvector_loader(cfg_path=cfg_path, M=M)
            dvector_loader = dvector_dataset.provide_test_original()
            assert M % 2 == 0
            enrollment_embeddings, verification_embeddings = torch.split(dvector_loader, int(dvector_loader.size(1) // 2), dim=1)

            num_speakers, num_utterances_enrollment, _ = enrollment_embeddings.shape
            _, num_utterances_verification, _ = verification_embeddings.shape

            # Calculate speaker models
            # If there's only one utterance for enrollment, use it directly; otherwise, calculate the mean
            if num_utterances_enrollment == 1:
                speaker_models = enrollment_embeddings.squeeze(1)
            else:
                speaker_models = torch.mean(enrollment_embeddings, dim=1)

            # Calculate similarities between speaker models and verification utterances
            similarities = torch.zeros((num_speakers, num_speakers * num_utterances_verification))
            labels = torch.zeros_like(similarities, dtype=torch.int)
            for i in range(num_speakers):
                for j in range(num_speakers):
                    for k in range(num_utterances_verification):
                        vec1 = speaker_models[i]
                        vec2 = verification_embeddings[j, k]
                        similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
                        similarities[i, j * num_utterances_verification + k] = similarity
                        if i == j:
                            labels[i, j * num_utterances_verification + k] = 1

            # Flatten similarity and label arrays
            similarities = similarities.flatten().numpy()
            labels = labels.flatten().numpy()

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(labels, similarities)

            # Calculate EER
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            final_eer.append(eer)

        final_eer = np.stack(final_eer)
        mean_EER = final_eer.mean()
        std_EER = final_eer.std()

        return mean_EER, std_EER, dvector_loader.shape[0]


    # below needed
    def EER_newmethod_epochy_anonymized(self, cfg_path, M=8, epochs=100):
        """
        evaluation (enrolment + verification)
        Open-set
        :epochs: because we are sampling each time, we have something like epoch here in testing
        """
        final_eer = []

        for _ in tqdm(range(epochs)):
            dvector_dataset = anonymized_dvector_loader(cfg_path=cfg_path, M=M)
            anonymized_embeddings, original_embeddings = dvector_dataset.provide_test_anonymized_and_original()

            # Split the original embeddings into enrollment and verification sets
            enrollment_set = original_embeddings[:, :original_embeddings.shape[1] // 2, :]
            verification_set_original = original_embeddings[:, original_embeddings.shape[1] // 2:, :]
            verification_set_anonymized = anonymized_embeddings[:, anonymized_embeddings.shape[1] // 2:, :]

            enrollment_embeddings = enrollment_set
            verification_embeddings = verification_set_anonymized


            num_speakers, num_utterances_enrollment, _ = enrollment_embeddings.shape
            _, num_utterances_verification, _ = verification_embeddings.shape

            # Calculate speaker models
            # If there's only one utterance for enrollment, use it directly; otherwise, calculate the mean
            if num_utterances_enrollment == 1:
                speaker_models = enrollment_embeddings.squeeze(1)
            else:
                speaker_models = torch.mean(enrollment_embeddings, dim=1)

            # Calculate similarities between speaker models and verification utterances
            similarities = torch.zeros((num_speakers, num_speakers * num_utterances_verification))
            labels = torch.zeros_like(similarities, dtype=torch.int)
            for i in range(num_speakers):
                for j in range(num_speakers):
                    for k in range(num_utterances_verification):
                        vec1 = speaker_models[i]
                        vec2 = verification_embeddings[j, k]
                        similarity = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
                        similarities[i, j * num_utterances_verification + k] = similarity
                        if i == j:
                            labels[i, j * num_utterances_verification + k] = 1

            # Flatten similarity and label arrays
            similarities = similarities.flatten().numpy()
            labels = labels.flatten().numpy()

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(labels, similarities)

            # Calculate EER
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
            final_eer.append(eer)

        final_eer = np.stack(final_eer)
        mean_EER = final_eer.mean()
        std_EER = final_eer.std()

        return mean_EER, std_EER, anonymized_embeddings.shape[0]
