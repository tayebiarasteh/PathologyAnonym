"""
classification_data.py
Created on Feb 5, 2024.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import glob
import os
import pdb
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import webrtcvad
import struct
from tqdm import tqdm
import random
from math import ceil, isnan
from scipy.ndimage.morphology import binary_dilation
import matplotlib.pyplot as plt
import math
import torch
from torch.utils.data import Dataset
from scipy.signal import get_window
from librosa.filters import mel
from scipy import signal

from config.serde import read_config

import warnings
warnings.filterwarnings('ignore')


# Global variables
int16_max = (2 ** 15) - 1
epsilon = 1e-15



class classification_data_preprocess():
    def __init__(self, cfg_path="/home/arasteh/Documents/Repositories/PathologyAnonym/config/config.yaml"):
        self.params = read_config(cfg_path)


    def main_org(self, file_path_input ="/cluster/arasteh/Documents/datasets/anonymization/PathologAnonym_project/masterlist_org.csv",
             ratio=0.1, exp_name='male', three_division=True):
        """main file after having a master csv, which divides to train, valid, test; and does preprocessing

        Parameters
        ----------
        ratio: float
            ratio of the split between train and (valid) and test.
            0.1 means 10% test, 10% validation, 80% training

        exp_name: str
            name of the experiment
        """

        train_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'train_' + exp_name + '.csv')
        valid_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'valid_' + exp_name + '.csv')
        test_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'test_' + exp_name + '.csv')

        statistics_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'statistics_' + exp_name + '.csv')

        selected_df = pd.read_csv(file_path_input, sep=';')

        selected_df = selected_df[selected_df['subset'] == 'adults']
        selected_df = selected_df[selected_df['automatic_WRR'] > 0]
        selected_df = selected_df[selected_df['age_y'] > 0]

        patient_df = selected_df[selected_df['mic_room'] == 'plantronics'] # means all the dysarthria (must be combined with adults only)
        # patient_df = selected_df[selected_df['mic_room'] == 'logitech'] # means all the dysphonia (must be combined with adults only)
        # patient_df = selected_df[selected_df['mic_room'] == 'maxillofacial'] # means all the dysglossia (must be combined with adults only)
        patient_df = patient_df[patient_df['patient_control'] == 'patient']

        # randomly reducing the number of patients to up to 2x of control
        patient_df = self.csv_reducing(patient_df, 160) # we have 81 adult controls

        control_df = selected_df[selected_df['mic_room'] == 'control_group_plantronics']
        control_df = control_df[control_df['patient_control'] == 'control']

        # len(control_df['speaker_id'].unique())

        if three_division:
            # creating train, valid, test csv files
            final_train_df_control, final_valid_df_control, final_test_df_control = self.train_valid_test_csv_creator_PEAKS(control_df, ratio)
            final_train_df_patient, final_valid_df_patient, final_test_df_patient = self.train_valid_test_csv_creator_PEAKS(patient_df, ratio)
        else:
            # creating train and test csv files
            final_train_df_control, final_test_df_control = self.train_test_csv_creator_PEAKS(control_df, ratio)
            final_train_df_patient, final_test_df_patient = self.train_test_csv_creator_PEAKS(patient_df, ratio)

        if three_division:
            final_valid_df = final_valid_df_patient.append(final_valid_df_control)
        final_train_df = final_train_df_patient.append(final_train_df_control)
        final_test_df = final_test_df_patient.append(final_test_df_control)

        # tisv preprocessing

        print('\ntisv preprocess\n')
        self.tisv_preproc(input_df=final_train_df, output_df_path=train_output_df_path, exp_name=exp_name)
        self.tisv_preproc(input_df=final_test_df, output_df_path=test_output_df_path, exp_name=exp_name)

        # saving histogram of the ages
        # train
        final_train_speaker_list = final_train_df['speaker_id'].unique().tolist()
        age_list_train = []
        for speaker in final_train_speaker_list:
            age_list_train.append(final_train_df[final_train_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(121)
        plt.hist(age_list_train)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("train | age: " + f"{np.mean(age_list_train):.1f}" + " +- " + f"{np.std(age_list_train):.1f}")

        # test
        final_test_speaker_list = final_test_df['speaker_id'].unique().tolist()
        age_list_test = []
        for speaker in final_test_speaker_list:
            age_list_test.append(final_test_df[final_test_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(122)
        plt.hist(age_list_test)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("test | age: " + f"{np.mean(age_list_test):.1f}" + " +- " + f"{np.std(age_list_test):.1f}")

        # plt.savefig(histogram_path)

        # statistics of the chosen dataset
        train_speaker_size = len(final_train_df['speaker_id'].unique().tolist())
        test_speaker_size = len(final_test_df['speaker_id'].unique().tolist())

        train_length = final_train_df['file_length'].sum() / 3600
        test_length = final_test_df['file_length'].sum() / 3600
        train_age_mean = final_train_df['age_y'].mean()
        test_age_mean = final_test_df['age_y'].mean()
        train_age_std = final_train_df['age_y'].std()
        test_age_std = final_test_df['age_y'].std()

        train_WA_mean = final_train_df[final_train_df['automatic_WA'] > -9999]['automatic_WA'].mean()
        train_WA_std = final_train_df[final_train_df['automatic_WA'] > -9999]['automatic_WA'].std()
        train_WR_mean = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        train_WR_std = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        test_WA_mean = final_test_df[final_test_df['automatic_WA'] > -9999]['automatic_WA'].mean()
        test_WA_std = final_test_df[final_test_df['automatic_WA'] > -9999]['automatic_WA'].std()
        test_WR_mean = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        test_WR_std = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        statistics_df = pd.DataFrame(columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std'])

        statistics_df = statistics_df.append(pd.DataFrame([['train', exp_name, train_speaker_size, train_length, train_age_mean, train_age_std,
                                                         train_WR_mean, train_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))
        statistics_df = statistics_df.append(pd.DataFrame([['test', exp_name, test_speaker_size, test_length, test_age_mean, test_age_std,
                                                         test_WR_mean, test_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))
        if three_division:
            valid_speaker_size = len(final_valid_df['speaker_id'].unique().tolist())
            valid_length = final_valid_df['file_length'].sum() / 3600
            valid_age_mean = final_valid_df['age_y'].mean()
            valid_age_std = final_valid_df['age_y'].std()

            valid_WA_mean = final_valid_df[final_valid_df['automatic_WA'] > -9999]['automatic_WA'].mean()
            valid_WA_std = final_valid_df[final_valid_df['automatic_WA'] > -9999]['automatic_WA'].std()
            valid_WR_mean = final_valid_df[final_valid_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
            valid_WR_std = final_valid_df[final_valid_df['automatic_WRR'] > 0]['automatic_WRR'].std()

            statistics_df = statistics_df.append(pd.DataFrame([['valid', exp_name, valid_speaker_size, valid_length, valid_age_mean, valid_age_std,
                                                                valid_WA_mean, valid_WA_std, valid_WR_mean, valid_WR_std]],
                             columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WA_mean', 'WA_std', 'WR_mean', 'WR_std']))

        statistics_df.to_csv(statistics_df_path, sep=';', index=False)


    def main_corresponding_anonymfiles(self, exp_name='male', old_exp_name='male'):
        """main file after having a master csv, which divides to train, valid, test; and does preprocessing

        Parameters
        ----------
        exp_name: str
            name of the experiment
        """

        train_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'train_' + exp_name + '.csv')
        valid_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'valid_' + exp_name + '.csv')
        test_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'test_' + exp_name + '.csv')

        statistics_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'statistics_' + exp_name + '.csv')

        train_path_input = os.path.join(self.params['file_path'], 'tisv_preprocess', old_exp_name, 'train_' + old_exp_name + '.csv')
        test_path_input = os.path.join(self.params['file_path'], 'tisv_preprocess', old_exp_name, 'test_' + old_exp_name + '.csv')

        final_train_df = pd.read_csv(train_path_input, sep=';')
        final_test_df = pd.read_csv(test_path_input, sep=';')

        # tisv preprocessing

        print('\ntisv preprocess\n')
        self.tisv_preproc_anonym(input_df=final_train_df, output_df_path=train_output_df_path, exp_name=exp_name, old_exp_name=old_exp_name)
        self.tisv_preproc_anonym(input_df=final_test_df, output_df_path=test_output_df_path, exp_name=exp_name, old_exp_name=old_exp_name)

        # saving histogram of the ages
        # train
        final_train_speaker_list = final_train_df['speaker_id'].unique().tolist()
        age_list_train = []
        for speaker in final_train_speaker_list:
            age_list_train.append(final_train_df[final_train_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(121)
        plt.hist(age_list_train)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("train | age: " + f"{np.mean(age_list_train):.1f}" + " +- " + f"{np.std(age_list_train):.1f}")

        # test
        final_test_speaker_list = final_test_df['speaker_id'].unique().tolist()
        age_list_test = []
        for speaker in final_test_speaker_list:
            age_list_test.append(final_test_df[final_test_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(122)
        plt.hist(age_list_test)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("test | age: " + f"{np.mean(age_list_test):.1f}" + " +- " + f"{np.std(age_list_test):.1f}")

        # plt.savefig(histogram_path)

        # statistics of the chosen dataset
        train_speaker_size = len(final_train_df['speaker_id'].unique().tolist())
        test_speaker_size = len(final_test_df['speaker_id'].unique().tolist())

        train_length = final_train_df['file_length'].sum() / 3600
        test_length = final_test_df['file_length'].sum() / 3600
        train_age_mean = final_train_df['age_y'].mean()
        test_age_mean = final_test_df['age_y'].mean()
        train_age_std = final_train_df['age_y'].std()
        test_age_std = final_test_df['age_y'].std()

        train_WR_mean = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        train_WR_std = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        test_WR_mean = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        test_WR_std = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        statistics_df = pd.DataFrame(columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std'])

        statistics_df = statistics_df.append(pd.DataFrame([['train', exp_name, train_speaker_size, train_length, train_age_mean, train_age_std,
                                                         train_WR_mean, train_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))
        statistics_df = statistics_df.append(pd.DataFrame([['test', exp_name, test_speaker_size, test_length, test_age_mean, test_age_std,
                                                         test_WR_mean, test_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))

        statistics_df.to_csv(statistics_df_path, sep=';', index=False)


    def train_valid_test_csv_creator_PEAKS(self, input_df, ratio):
        """splits the csv file to train, valid, and test csv files
        Parameters
        ----------
        """
        # initiating valid and train dfs
        final_train_data = pd.DataFrame([])
        final_valid_data = pd.DataFrame([])
        final_test_data = pd.DataFrame([])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)
        val_num = ceil(len(PEAKS_speaker_list) / (1/ratio))

        # take X% of PEAKS speakers as validation
        val_speakers = PEAKS_speaker_list[:val_num]
        # take X% of PEAKS speakers as test
        test_speakers = PEAKS_speaker_list[val_num:2*val_num]
        # take rest of PEAKS speakers as training
        train_speakers = PEAKS_speaker_list[2*val_num:]

        # adding PEAKS files to valid
        for speaker in val_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_valid_data = final_valid_data.append(selected_speaker_df)

        # adding PEAKS files to test
        for speaker in test_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_test_data = final_test_data.append(selected_speaker_df)

        # adding PEAKS files to train
        for speaker in train_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_train_data = final_train_data.append(selected_speaker_df)

        # sort based on speaker id
        final_train_data = final_train_data.sort_values(['relative_path'])
        final_valid_data = final_valid_data.sort_values(['relative_path'])
        final_test_data = final_test_data.sort_values(['relative_path'])

        return final_train_data, final_valid_data, final_test_data



    def train_test_csv_creator_PEAKS(self, input_df, ratio):
        """splits the csv file to train, valid, and test csv files
        Parameters
        ----------
        """
        # initiating valid and train dfs
        final_train_data = pd.DataFrame([])
        final_test_data = pd.DataFrame([])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)
        test_num = ceil(len(PEAKS_speaker_list) / (1/ratio))

        # take X% of PEAKS speakers as test
        test_speakers = PEAKS_speaker_list[:test_num]
        # take rest of PEAKS speakers as training
        train_speakers = PEAKS_speaker_list[test_num:]

        # adding PEAKS files to test
        for speaker in test_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_test_data = final_test_data.append(selected_speaker_df)

        # adding PEAKS files to train
        for speaker in train_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_train_data = final_train_data.append(selected_speaker_df)

        # sort based on speaker id
        final_train_data = final_train_data.sort_values(['relative_path'])
        final_test_data = final_test_data.sort_values(['relative_path'])

        return final_train_data, final_test_data


    def csv_reducing(self, input_df, max_num):
        """
        Parameters
        ----------
        """
        final_data = pd.DataFrame([])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)

        # take X% of PEAKS speakers
        try:
            new_speakers = PEAKS_speaker_list[:max_num]
        except:
            new_speakers = PEAKS_speaker_list

        # adding PEAKS files to test
        for speaker in new_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])

        return final_data



    def trim_long_silences(self, wav):
        """
        Ensures that segments without voice in the waveform remain no longer than a
        threshold determined by the VAD parameters in params.py.

        Parameters
        ----------
        wav: numpy array of floats
            the raw waveform as a numpy array of floats

        Returns
        -------
        trimmed_wav: numpy array of floats
            the same waveform with silences trimmed
            away (length <= original wav length)
        """

        # Compute the voice detection window size
        samples_per_window = (self.params['preprocessing']['vad_window_length'] * self.params['preprocessing']['sr']) // 1000

        # Trim the end of the audio to have a multiple of the window size
        wav = wav[:len(wav) - (len(wav) % samples_per_window)]

        # Convert the float waveform to 16-bit mono PCM
        pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

        # Perform voice activation detection
        voice_flags = []
        vad = webrtcvad.Vad(mode=3)
        for window_start in range(0, len(wav), samples_per_window):
            window_end = window_start + samples_per_window
            voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                             sample_rate=self.params['preprocessing']['sr']))
        voice_flags = np.array(voice_flags)

        # Smooth the voice detection with a moving average
        def moving_average(array, width):
            array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
            ret = np.cumsum(array_padded, dtype=float)
            ret[width:] = ret[width:] - ret[:-width]
            return ret[width - 1:] / width

        audio_mask = moving_average(voice_flags, self.params['preprocessing']['vad_moving_average_width'])
        audio_mask = np.round(audio_mask).astype(np.bool)

        # Dilate the voiced regions
        audio_mask = binary_dilation(audio_mask, np.ones(self.params['preprocessing']['vad_max_silence_length'] + 1))
        audio_mask = np.repeat(audio_mask, samples_per_window)

        return wav[audio_mask == True]


    def normalize_volume(self, wav, target_dBFS, increase_only=False, decrease_only=False):
        if increase_only and decrease_only:
            raise ValueError("Both increase only and decrease only are set")
        rms = np.sqrt(np.mean((wav * int16_max) ** 2))
        wave_dBFS = 20 * np.log10(rms / int16_max)
        dBFS_change = target_dBFS - wave_dBFS
        if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
            return wav
        return wav * (10 ** (dBFS_change / 20))



    def tisv_preproc(self, input_df, output_df_path, exp_name):
        """
        GE2E-loss-based pre-processing of validation & test utterances for text-independent speaker verification.
        References:
            https://github.com/JanhHyun/Speaker_Verification/
            https://github.com/HarryVolek/PyTorch_Speaker_Verification/
            https://github.com/resemble-ai/Resemblyzer/
        """

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame([])

        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):

            utter_path = os.path.join(self.params['file_path'], row['relative_path'])
            utter, sr = sf.read(utter_path)

            # pre-processing and voice activity detection (VAD) part 1
            utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'], increase_only=True)
            utter = self.trim_long_silences(utter)
            # if utter.shape[0] < utter_min_len:
            #     continue

            # basically this does nothing if 60 is chosen, 60 is too high, so the whole wav will be selected.
            # This just makes an interval from beginning to the end.
            intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection part 2

            for interval_index, interval in enumerate(intervals):
                # if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficiently long,
                utter_part = utter[interval[0]:interval[1]]

                # concatenate all the partial utterances of each utterance
                if interval_index == 0:
                    utter_whole = utter_part
                else:
                    try:
                        utter_whole = np.hstack((utter_whole, utter_part))
                    except:
                        utter_whole = utter_part
            if 'utter_whole' in locals():

                S = librosa.core.stft(y=utter_whole, n_fft=self.params['preprocessing']['nfft'],
                                      win_length=int(self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                      hop_length=int(self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
                S = np.abs(S) ** 2
                mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'], n_fft=self.params['preprocessing']['nfft'],
                                                n_mels=self.params['preprocessing']['nmels'])

                SS = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of partial utterance
                os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                         os.path.dirname(row['relative_path'])), exist_ok=True)

                rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, os.path.dirname(row['relative_path']),
                                        os.path.basename(row['relative_path']).replace('.wav', '.npy'))
                np.save(rel_path, SS)

                # add to the new dataframe
                tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                    os.path.dirname(row['relative_path']),
                                                        os.path.basename(row['relative_path']).replace('.wav', '.npy')),
                                       row['speaker_id'], row['subset'], row['gender'], row['location'], row['age_y'],
                                       row['microphone'], row['patient_control'], row['automatic_WRR'], utter_whole.shape[0] / self.params['preprocessing']['sr'],
                                       row['user_id'], row['session'], row['father_tongue'], row['mother_tongue'], row['test_type'], row['mic_room'], row['diagnosis']
                                        ]],
                                     columns=['relative_path', 'speaker_id', 'subset', 'gender', 'location', 'age_y', 'microphone',
                                              'patient_control', 'automatic_WRR', 'file_length', 'user_id', 'session', 'father_tongue',
                                              'mother_tongue', 'test_type', 'mic_room', 'diagnosis'])
                final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
         # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)



    def tisv_preproc_anonym(self, input_df, output_df_path, exp_name, old_exp_name):
        """
        GE2E-loss-based pre-processing of validation & test utterances for text-independent speaker verification.
        References:
            https://github.com/JanhHyun/Speaker_Verification/
            https://github.com/HarryVolek/PyTorch_Speaker_Verification/
            https://github.com/resemble-ai/Resemblyzer/
        """

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame([])

        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):

            row_relative_path = row['relative_path'].replace('tisv_preprocess/' + old_exp_name + '/PEAKS', 'PEAKS_anonymized')
            row_relative_path = row_relative_path.replace('.npy', '.wav')

            utter_path = os.path.join(self.params['file_path'], row_relative_path)
            utter, sr = sf.read(utter_path)

            # pre-processing and voice activity detection (VAD) part 1
            utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'], increase_only=True)
            utter = self.trim_long_silences(utter)
            # if utter.shape[0] < utter_min_len:
            #     continue

            # basically this does nothing if 60 is chosen, 60 is too high, so the whole wav will be selected.
            # This just makes an interval from beginning to the end.
            intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection part 2

            for interval_index, interval in enumerate(intervals):
                # if (interval[1] - interval[0]) > utter_min_len:  # If partial utterance is sufficiently long,
                utter_part = utter[interval[0]:interval[1]]

                # concatenate all the partial utterances of each utterance
                if interval_index == 0:
                    utter_whole = utter_part
                else:
                    try:
                        utter_whole = np.hstack((utter_whole, utter_part))
                    except:
                        utter_whole = utter_part
            if 'utter_whole' in locals():

                S = librosa.core.stft(y=utter_whole, n_fft=self.params['preprocessing']['nfft'],
                                      win_length=int(self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                      hop_length=int(self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
                S = np.abs(S) ** 2
                mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'], n_fft=self.params['preprocessing']['nfft'],
                                                n_mels=self.params['preprocessing']['nmels'])

                SS = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of partial utterance
                os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                         os.path.dirname(row_relative_path)), exist_ok=True)

                rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, os.path.dirname(row_relative_path),
                                        os.path.basename(row_relative_path).replace('.wav', '.npy'))
                np.save(rel_path, SS)

                # add to the new dataframe
                tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                    os.path.dirname(row_relative_path),
                                                        os.path.basename(row_relative_path).replace('.wav', '.npy')),
                                       row['speaker_id'], row['subset'], row['gender'], row['location'], row['age_y'],
                                       row['microphone'], row['patient_control'], row['automatic_WRR'], utter_whole.shape[0] / self.params['preprocessing']['sr'],
                                       row['user_id'], row['session'], row['father_tongue'], row['mother_tongue'], row['test_type'], row['mic_room'], row['diagnosis']
                                        ]],
                                     columns=['relative_path', 'speaker_id', 'subset', 'gender', 'location', 'age_y', 'microphone',
                                              'patient_control', 'automatic_WRR', 'file_length', 'user_id', 'session', 'father_tongue',
                                              'mother_tongue', 'test_type', 'mic_room', 'diagnosis'])
                final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
         # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)


    def csv_speaker_trimmer(self, input_df):
        """only keeps the speakers which have at least 8 utterances
        Parameters
        ----------
        Returns
        ----------
        """
        final_data = pd.DataFrame([])

        list_speakers = input_df['speaker_id'].unique().tolist()

        for speaker in list_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            if len(selected_speaker_df) >= 8:
                final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])
        return final_data




class classification_tisvcontentbased_data_preprocess():
    def __init__(self, cfg_path="/home/arasteh/Documents/Repositories/PathologyAnonym/config/config.yaml"):
        self.params = read_config(cfg_path)


    def main_org(self, file_path_input ="/cluster/arasteh/Documents/datasets/anonymization/PathologAnonym_project/masterlist_org.csv",
             ratio=0.1, exp_name='male', three_division=True):
        """main file after having a master csv, which divides to train, valid, test; and does preprocessing

        Parameters
        ----------
        ratio: float
            ratio of the split between train and (valid) and test.
            0.1 means 10% test, 10% validation, 80% training

        exp_name: str
            name of the experiment
        """

        train_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'train_' + exp_name + '.csv')
        valid_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'valid_' + exp_name + '.csv')
        test_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'test_' + exp_name + '.csv')

        statistics_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'statistics_' + exp_name + '.csv')

        selected_df = pd.read_csv(file_path_input, sep=';')

        selected_df = selected_df[selected_df['subset'] == 'adults']
        selected_df = selected_df[selected_df['automatic_WRR'] > 0]
        selected_df = selected_df[selected_df['age_y'] > 0]

        patient_df = selected_df[selected_df['mic_room'] == 'plantronics'] # means all the dysarthria (must be combined with adults only)
        # patient_df = selected_df[selected_df['mic_room'] == 'logitech'] # means all the dysphonia (must be combined with adults only)
        # patient_df = selected_df[selected_df['mic_room'] == 'maxillofacial'] # means all the dysglossia (must be combined with adults only)
        patient_df = patient_df[patient_df['patient_control'] == 'patient']

        # randomly reducing the number of patients to up to 2x of control
        patient_df = self.csv_reducing(patient_df, 160) # we have 81 adult controls

        control_df = selected_df[selected_df['mic_room'] == 'control_group_plantronics']
        control_df = control_df[control_df['patient_control'] == 'control']

        # len(control_df['speaker_id'].unique())

        if three_division:
            # creating train, valid, test csv files
            final_train_df_control, final_valid_df_control, final_test_df_control = self.train_valid_test_csv_creator_PEAKS(control_df, ratio)
            final_train_df_patient, final_valid_df_patient, final_test_df_patient = self.train_valid_test_csv_creator_PEAKS(patient_df, ratio)
        else:
            # creating train and test csv files
            final_train_df_control, final_test_df_control = self.train_test_csv_creator_PEAKS(control_df, ratio)
            final_train_df_patient, final_test_df_patient = self.train_test_csv_creator_PEAKS(patient_df, ratio)

        if three_division:
            final_valid_df = final_valid_df_patient.append(final_valid_df_control)
        final_train_df = final_train_df_patient.append(final_train_df_control)
        final_test_df = final_test_df_patient.append(final_test_df_control)

        # tisv preprocessing

        print('\ntisv preprocess\n')
        self.get_mel_content(input_df=final_train_df, output_df_path=train_output_df_path, exp_name=exp_name)
        self.get_mel_content(input_df=final_test_df, output_df_path=test_output_df_path, exp_name=exp_name)

        # saving histogram of the ages
        # train
        final_train_speaker_list = final_train_df['speaker_id'].unique().tolist()
        age_list_train = []
        for speaker in final_train_speaker_list:
            age_list_train.append(final_train_df[final_train_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(121)
        plt.hist(age_list_train)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("train | age: " + f"{np.mean(age_list_train):.1f}" + " +- " + f"{np.std(age_list_train):.1f}")

        # test
        final_test_speaker_list = final_test_df['speaker_id'].unique().tolist()
        age_list_test = []
        for speaker in final_test_speaker_list:
            age_list_test.append(final_test_df[final_test_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(122)
        plt.hist(age_list_test)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("test | age: " + f"{np.mean(age_list_test):.1f}" + " +- " + f"{np.std(age_list_test):.1f}")

        # plt.savefig(histogram_path)

        # statistics of the chosen dataset
        train_speaker_size = len(final_train_df['speaker_id'].unique().tolist())
        test_speaker_size = len(final_test_df['speaker_id'].unique().tolist())

        train_length = final_train_df['file_length'].sum() / 3600
        test_length = final_test_df['file_length'].sum() / 3600
        train_age_mean = final_train_df['age_y'].mean()
        test_age_mean = final_test_df['age_y'].mean()
        train_age_std = final_train_df['age_y'].std()
        test_age_std = final_test_df['age_y'].std()

        train_WR_mean = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        train_WR_std = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        test_WR_mean = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        test_WR_std = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        statistics_df = pd.DataFrame(columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std'])

        statistics_df = statistics_df.append(pd.DataFrame([['train', exp_name, train_speaker_size, train_length, train_age_mean, train_age_std,
                                                         train_WR_mean, train_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))
        statistics_df = statistics_df.append(pd.DataFrame([['test', exp_name, test_speaker_size, test_length, test_age_mean, test_age_std,
                                                         test_WR_mean, test_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))
        if three_division:
            valid_speaker_size = len(final_valid_df['speaker_id'].unique().tolist())
            valid_length = final_valid_df['file_length'].sum() / 3600
            valid_age_mean = final_valid_df['age_y'].mean()
            valid_age_std = final_valid_df['age_y'].std()

            valid_WA_mean = final_valid_df[final_valid_df['automatic_WA'] > -9999]['automatic_WA'].mean()
            valid_WA_std = final_valid_df[final_valid_df['automatic_WA'] > -9999]['automatic_WA'].std()
            valid_WR_mean = final_valid_df[final_valid_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
            valid_WR_std = final_valid_df[final_valid_df['automatic_WRR'] > 0]['automatic_WRR'].std()

            statistics_df = statistics_df.append(pd.DataFrame([['valid', exp_name, valid_speaker_size, valid_length, valid_age_mean, valid_age_std,
                                                                valid_WA_mean, valid_WA_std, valid_WR_mean, valid_WR_std]],
                             columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WA_mean', 'WA_std', 'WR_mean', 'WR_std']))

        statistics_df.to_csv(statistics_df_path, sep=';', index=False)


    def main_corresponding_anonymfiles(self, exp_name='male', old_exp_name='male'):
        """main file after having a master csv, which divides to train, valid, test; and does preprocessing

        Parameters
        ----------
        exp_name: str
            name of the experiment
        """

        train_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'train_' + exp_name + '.csv')
        valid_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'valid_' + exp_name + '.csv')
        test_output_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'test_' + exp_name + '.csv')

        statistics_df_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name, 'statistics_' + exp_name + '.csv')

        train_path_input = os.path.join(self.params['file_path'], 'tisv_preprocess', old_exp_name, 'train_' + old_exp_name + '.csv')
        test_path_input = os.path.join(self.params['file_path'], 'tisv_preprocess', old_exp_name, 'test_' + old_exp_name + '.csv')

        final_train_df = pd.read_csv(train_path_input, sep=';')
        final_test_df = pd.read_csv(test_path_input, sep=';')

        # tisv preprocessing

        print('\ntisv preprocess\n')
        self.get_mel_content_anonym(input_df=final_train_df, output_df_path=train_output_df_path, exp_name=exp_name, old_exp_name=old_exp_name)
        self.get_mel_content_anonym(input_df=final_test_df, output_df_path=test_output_df_path, exp_name=exp_name, old_exp_name=old_exp_name)

        # saving histogram of the ages
        # train
        final_train_speaker_list = final_train_df['speaker_id'].unique().tolist()
        age_list_train = []
        for speaker in final_train_speaker_list:
            age_list_train.append(final_train_df[final_train_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(121)
        plt.hist(age_list_train)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("train | age: " + f"{np.mean(age_list_train):.1f}" + " +- " + f"{np.std(age_list_train):.1f}")

        # test
        final_test_speaker_list = final_test_df['speaker_id'].unique().tolist()
        age_list_test = []
        for speaker in final_test_speaker_list:
            age_list_test.append(final_test_df[final_test_df['speaker_id'] == speaker]['age_y'].mean())
        plt.subplot(122)
        plt.hist(age_list_test)
        plt.xlabel('Age [years]')
        plt.ylabel('Num Speakers')
        plt.title("test | age: " + f"{np.mean(age_list_test):.1f}" + " +- " + f"{np.std(age_list_test):.1f}")

        # plt.savefig(histogram_path)

        # statistics of the chosen dataset
        train_speaker_size = len(final_train_df['speaker_id'].unique().tolist())
        test_speaker_size = len(final_test_df['speaker_id'].unique().tolist())

        train_length = final_train_df['file_length'].sum() / 3600
        test_length = final_test_df['file_length'].sum() / 3600
        train_age_mean = final_train_df['age_y'].mean()
        test_age_mean = final_test_df['age_y'].mean()
        train_age_std = final_train_df['age_y'].std()
        test_age_std = final_test_df['age_y'].std()

        train_WR_mean = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        train_WR_std = final_train_df[final_train_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        test_WR_mean = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].mean()
        test_WR_std = final_test_df[final_test_df['automatic_WRR'] > 0]['automatic_WRR'].std()

        statistics_df = pd.DataFrame(columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std'])

        statistics_df = statistics_df.append(pd.DataFrame([['train', exp_name, train_speaker_size, train_length, train_age_mean, train_age_std,
                                                         train_WR_mean, train_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))
        statistics_df = statistics_df.append(pd.DataFrame([['test', exp_name, test_speaker_size, test_length, test_age_mean, test_age_std,
                                                         test_WR_mean, test_WR_std]],
                     columns=['subset', 'exp_name', 'total_speakers', 'total_hours', 'age_mean', 'age_std', 'WR_mean', 'WR_std']))

        statistics_df.to_csv(statistics_df_path, sep=';', index=False)


    def get_mel_content(self, input_df, output_df_path, exp_name):
        """
        References:
            https://github.com/RF5/simple-autovc/
            https://github.com/CODEJIN/AutoVC/

        Parameters
        ----------
        """

        mel_basis_hifi = mel(self.params['preprocessing']['sr'], 1024, fmin=0, fmax=8000, n_mels=80).T
        b, a = self.butter_highpass(30, self.params['preprocessing']['sr'], order=5)

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame([])

        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):

            # Read audio file
            utter_path = os.path.join(self.params['file_path'], row['relative_path'])
            x, fs = sf.read(utter_path)

            if x.shape[0] < utter_min_len:
                continue

            x = librosa.resample(x, fs, self.params['preprocessing']['sr'])
            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0]) - 0.5) * 1e-06
            # Compute spect
            D = self.pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis_hifi)
            S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

            os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                     os.path.dirname(row['relative_path'])), exist_ok=True)

            rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                    os.path.dirname(row['relative_path']),
                                    os.path.basename(row['relative_path']).replace('.wav', '.npy'))
            S = S.transpose(1,0)
            np.save(rel_path, S)

            # add to the new dataframe
            tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                os.path.dirname(row['relative_path']),
                                                os.path.basename(row['relative_path']).replace('.wav', '.npy')),
                                   row['speaker_id'], row['subset'], row['gender'], row['location'], row['age_y'],
                                   row['microphone'], row['patient_control'], row['automatic_WRR'],
                                   x.shape[0] / self.params['preprocessing']['sr'],
                                   row['user_id'], row['session'], row['father_tongue'], row['mother_tongue'],
                                   row['test_type'], row['mic_room'], row['diagnosis']
                                   ]],
                                 columns=['relative_path', 'speaker_id', 'subset', 'gender', 'location', 'age_y',
                                          'microphone',
                                          'patient_control', 'automatic_WRR', 'file_length', 'user_id', 'session',
                                          'father_tongue',
                                          'mother_tongue', 'test_type', 'mic_room', 'diagnosis'])
            final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
        # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)


    def get_mel_content_anonym(self, input_df, output_df_path, exp_name, old_exp_name):
        """
        References:
            https://github.com/RF5/simple-autovc/
            https://github.com/CODEJIN/AutoVC/

        Parameters
        ----------
        """

        mel_basis_hifi = mel(self.params['preprocessing']['sr'], 1024, fmin=0, fmax=8000, n_mels=80).T
        b, a = self.butter_highpass(30, self.params['preprocessing']['sr'], order=5)

        # lower bound of utterance length
        utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']

        final_dataframe = pd.DataFrame([])

        for index, row in tqdm(input_df.iterrows(), total=input_df.shape[0]):

            row_relative_path = row['relative_path'].replace('tisv_preprocess/' + old_exp_name + '/PEAKS',
                                                             'PEAKS_anonymized')
            row_relative_path = row_relative_path.replace('.npy', '.wav')

            # Read audio file
            utter_path = os.path.join(self.params['file_path'], row_relative_path)
            x, fs = sf.read(utter_path)

            if x.shape[0] < utter_min_len:
                continue

            x = librosa.resample(x, fs, self.params['preprocessing']['sr'])
            # Remove drifting noise
            y = signal.filtfilt(b, a, x)
            # Ddd a little random noise for model roubstness
            wav = y * 0.96 + (np.random.RandomState().rand(y.shape[0]) - 0.5) * 1e-06
            # Compute spect
            D = self.pySTFT(wav).T
            # Convert to mel and normalize
            D_mel = np.dot(D, mel_basis_hifi)
            S = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

            os.makedirs(os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                     os.path.dirname(row_relative_path)), exist_ok=True)

            rel_path = os.path.join(self.params['file_path'], 'tisv_preprocess', exp_name,
                                    os.path.dirname(row_relative_path),
                                    os.path.basename(row_relative_path).replace('.wav', '.npy'))
            S = S.transpose(1,0)
            np.save(rel_path, S)

            # add to the new dataframe
            tempp = pd.DataFrame([[os.path.join('tisv_preprocess', exp_name,
                                                os.path.dirname(row_relative_path),
                                                os.path.basename(row_relative_path).replace('.wav', '.npy')),
                                   row['speaker_id'], row['subset'], row['gender'], row['location'], row['age_y'],
                                   row['microphone'], row['patient_control'], row['automatic_WRR'],
                                   x.shape[0] / self.params['preprocessing']['sr'],
                                   row['user_id'], row['session'], row['father_tongue'], row['mother_tongue'],
                                   row['test_type'], row['mic_room'], row['diagnosis']
                                   ]],
                                 columns=['relative_path', 'speaker_id', 'subset', 'gender', 'location', 'age_y',
                                          'microphone',
                                          'patient_control', 'automatic_WRR', 'file_length', 'user_id', 'session',
                                          'father_tongue',
                                          'mother_tongue', 'test_type', 'mic_room', 'diagnosis'])
            final_dataframe = final_dataframe.append(tempp)

        # to check the criterion of 8 utters after VAD
        final_dataframe = self.csv_speaker_trimmer(final_dataframe)
        # sort based on speaker id
        final_data = final_dataframe.sort_values(['relative_path'])
        final_data.to_csv(output_df_path, sep=';', index=False)


    def train_valid_test_csv_creator_PEAKS(self, input_df, ratio):
        """splits the csv file to train, valid, and test csv files
        Parameters
        ----------
        """
        # initiating valid and train dfs
        final_train_data = pd.DataFrame([])
        final_valid_data = pd.DataFrame([])
        final_test_data = pd.DataFrame([])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)
        val_num = ceil(len(PEAKS_speaker_list) / (1/ratio))

        # take X% of PEAKS speakers as validation
        val_speakers = PEAKS_speaker_list[:val_num]
        # take X% of PEAKS speakers as test
        test_speakers = PEAKS_speaker_list[val_num:2*val_num]
        # take rest of PEAKS speakers as training
        train_speakers = PEAKS_speaker_list[2*val_num:]

        # adding PEAKS files to valid
        for speaker in val_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_valid_data = final_valid_data.append(selected_speaker_df)

        # adding PEAKS files to test
        for speaker in test_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_test_data = final_test_data.append(selected_speaker_df)

        # adding PEAKS files to train
        for speaker in train_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_train_data = final_train_data.append(selected_speaker_df)

        # sort based on speaker id
        final_train_data = final_train_data.sort_values(['relative_path'])
        final_valid_data = final_valid_data.sort_values(['relative_path'])
        final_test_data = final_test_data.sort_values(['relative_path'])

        return final_train_data, final_valid_data, final_test_data



    def train_test_csv_creator_PEAKS(self, input_df, ratio):
        """splits the csv file to train, valid, and test csv files
        Parameters
        ----------
        """
        # initiating valid and train dfs
        final_train_data = pd.DataFrame([])
        final_test_data = pd.DataFrame([])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)
        test_num = ceil(len(PEAKS_speaker_list) / (1/ratio))

        # take X% of PEAKS speakers as test
        test_speakers = PEAKS_speaker_list[:test_num]
        # take rest of PEAKS speakers as training
        train_speakers = PEAKS_speaker_list[test_num:]

        # adding PEAKS files to test
        for speaker in test_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_test_data = final_test_data.append(selected_speaker_df)

        # adding PEAKS files to train
        for speaker in train_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_train_data = final_train_data.append(selected_speaker_df)

        # sort based on speaker id
        final_train_data = final_train_data.sort_values(['relative_path'])
        final_test_data = final_test_data.sort_values(['relative_path'])

        return final_train_data, final_test_data


    def csv_reducing(self, input_df, max_num):
        """
        Parameters
        ----------
        """
        final_data = pd.DataFrame([])

        PEAKS_speaker_list = input_df['speaker_id'].unique().tolist()
        random.shuffle(PEAKS_speaker_list)

        # take X% of PEAKS speakers
        try:
            new_speakers = PEAKS_speaker_list[:max_num]
        except:
            new_speakers = PEAKS_speaker_list

        # adding PEAKS files to test
        for speaker in new_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])

        return final_data


    def pad_seq(self, x, base=32):
        len_out = int(base * math.ceil(float(x.shape[0]) / base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return torch.nn.functional.pad(x, (0, 0, 0, len_pad), value=0), len_pad

    def butter_highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def pySTFT(self, x, fft_length=1024, hop_length=256):
        x = np.pad(x, int(fft_length // 2), mode='reflect')
        noverlap = fft_length - hop_length
        shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
        strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
        result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        fft_window = get_window('hann', fft_length, fftbins=True)
        result = np.fft.rfft(fft_window * result, n=fft_length).T
        return np.abs(result)



    def csv_speaker_trimmer(self, input_df):
        """only keeps the speakers which have at least 8 utterances
        Parameters
        ----------
        Returns
        ----------
        """
        final_data = pd.DataFrame([])

        list_speakers = input_df['speaker_id'].unique().tolist()

        for speaker in list_speakers:
            selected_speaker_df = input_df[input_df['speaker_id'] == speaker]
            if len(selected_speaker_df) >= 8:
                final_data = final_data.append(selected_speaker_df)

        # sort based on speaker id
        final_data = final_data.sort_values(['relative_path'])
        return final_data






class Dataloader_disorder(Dataset):
    """
    This is the pipeline based on Pytorch's Dataset and Dataloader
    """
    def __init__(self, cfg_path, mode='train', experiment_name='name'):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        mode: str
            Nature of operation to be done with the data.
                Possible inputs are train, valid, test
                Default value: train
        """

        self.cfg_path = cfg_path
        self.params = read_config(cfg_path)
        self.file_base_dir = self.params['file_path']
        self.sampling_val = 180

        if mode == 'train':
            self.main_df = pd.read_csv(os.path.join(self.file_base_dir, "tisv_preprocess", experiment_name, "train_" + experiment_name + ".csv"), sep=';')
        elif mode == 'valid':
            self.main_df = pd.read_csv(os.path.join(self.file_base_dir, "tisv_preprocess", experiment_name, "test_" + experiment_name + ".csv"), sep=';')
        elif mode == 'test':
            self.main_df = pd.read_csv(os.path.join(self.file_base_dir, "tisv_preprocess", experiment_name, "test_" + experiment_name + ".csv"), sep=';')

        # self.main_df = self.main_df[self.main_df['file_length'] > 1.85] # to have at least 180 points (for the tisv method)
        self.main_df = self.main_df[self.main_df['file_length'] > 3] # to have at least 180 points (for the content based mel method)

        self.speaker_list = self.main_df['speaker_id'].unique().tolist()




    def __len__(self):
        """Returns the length of the dataset"""
        return len(self.speaker_list)


    def __getitem__(self, idx):
        """
        Parameters
        ----------
        idx: int

        Returns
        -------
        img: torch tensor
        label: torch tensor
        """
        output_tensor = []

        # select a speaker
        selected_speaker = self.speaker_list[idx]
        selected_speaker_df = self.main_df[self.main_df['speaker_id'] == selected_speaker]

        # randomly select M utterances from the speaker
        shuff_selected_speaker_df = selected_speaker_df.sample(frac=1).reset_index(drop=True)

        shuff_selected_speaker_df = shuff_selected_speaker_df[:self.params['Network']['M']]

        # return M utterances
        for index, row in shuff_selected_speaker_df.iterrows():
            # select a random utterance
            utterance = np.load(os.path.join(self.file_base_dir, row['relative_path']))

            # randomly sample a fixed specified length
            id = np.random.randint(0, utterance.shape[1] - self.sampling_val, 1)
            utterance = utterance[:, id[0]:id[0] + self.sampling_val]

            output_tensor.append(utterance)

        output_tensor = np.stack((output_tensor, output_tensor, output_tensor), axis=1) # (n=M, c=3, h=melsize, w=sampling_val)
        output_tensor = torch.from_numpy(output_tensor) # (M, c, h) treated as (n, h, w)


        # one hot
        if shuff_selected_speaker_df['patient_control'].values[0] == 'patient':
            label = torch.ones((self.params['Network']['M']), 2)
            label[:, 0] = 0
        elif shuff_selected_speaker_df['patient_control'].values[0] == 'control':
            label = torch.zeros((self.params['Network']['M']), 2)
            label[:, 0] = 1

        label = label.float()

        return output_tensor, label



    def pos_weight(self):
        """
        Calculates a weight for positive examples for each class and returns it as a tensor
        Only using the training set.
        """

        full_length = len(self.main_df)

        disease_length = sum(self.main_df['patient_control'].values == 'patient')
        output_tensor = (full_length - disease_length) / (disease_length + epsilon)

        output_tensor = torch.Tensor([output_tensor])
        return output_tensor





if __name__ == '__main__':
    # handler = classification_data_preprocess(cfg_path="/home/soroosh/Documents/Repositories/PathologyAnonym/config/config.yaml")
    handler = classification_tisvcontentbased_data_preprocess(cfg_path="/home/soroosh/Documents/Repositories/PathologyAnonym/config/config.yaml")

    handler.main_org(file_path_input ="/home/soroosh/Documents/datasets/anonymization/PathologAnonym_project/masterlist_org.csv",
             ratio=0.3, exp_name='dysarthria_70_30_contentmel', three_division=False)

    handler.main_corresponding_anonymfiles(exp_name='dysarthria_70_30_contentmel_anonym', old_exp_name='dysarthria_70_30_contentmel')