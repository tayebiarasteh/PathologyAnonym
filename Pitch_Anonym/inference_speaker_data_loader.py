"""
inference_speaker_data_loader.py
Created on Oct 31, 2023.
Data loader.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""

import numpy as np
import torch
import os
import pdb
import glob
import random
import pandas as pd
import soundfile as sf
import webrtcvad
import struct
from tqdm import tqdm
import librosa
from scipy.ndimage.morphology import binary_dilation
from speechbrain.pretrained import HIFIGAN
from scipy.io.wavfile import write
from librosa.filters import mel
from scipy import signal
import math
from scipy.signal import get_window
import noisereduce as nr

from config.serde import read_config


epsilon = 1e-15
int16_max = (2 ** 15) - 1






class loader_for_dvector_creation:
    def __init__(self, cfg_path='./config/config.json', spk_nmels=40):
        """For d-vector creation (prediction of the input utterances) step.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.file_path = self.params['file_path']
        self.utter_min_len = (self.params['preprocessing']['tisv_frame'] * self.params['preprocessing']['hop'] +
                         self.params['preprocessing']['window']) * self.params['preprocessing']['sr']
        self.nmels = spk_nmels
        self.main_df = pd.read_csv(os.path.join(self.params['file_path'], "PathologAnonym_project/masterlist_org.csv"), sep=';')


    def provide_data_original(self):
        """
        Returns
        -------
        speakers: dictionary of list
            a dictionary of all the speakers. Each speaker contains a list of
            all its utterances converted to mel spectrograms
        """
        # dictionary of speakers
        speakers = {}
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()

        for speaker_name in tqdm(self.speaker_list):
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]
            # list of utterances of each speaker
            utterances = []

            for index, row in selected_speaker_df.iterrows():
                utter, sr = sf.read(os.path.join(self.file_path, row['relative_path']))
                utterance = self.tisv_preproc(utter)
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances

        return speakers


    def provide_data_anonymized(self):
        """
        Returns
        -------
        speakers: dictionary of list
            a dictionary of all the speakers. Each speaker contains a list of
            all its utterances converted to mel spectrograms
        """
        # dictionary of speakers
        speakers = {}
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()

        for speaker_name in tqdm(self.speaker_list):
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]
            # list of utterances of each speaker
            utterances = []

            for index, row in selected_speaker_df.iterrows():
                path = os.path.join(self.file_path, row['relative_path'])
                path_anonymized = path.replace('/PEAKS', '/PEAKS_Pitch_anonymized')
                try:
                    utter, sr = sf.read(path_anonymized)
                except:
                    continue
                utterance = self.tisv_preproc(utter)
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances

        return speakers


    def tisv_preproc(self, utter):
        """
        GE2E-loss-based pre-processing
        References:
            https://github.com/JanhHyun/Speaker_Verification/
            https://github.com/HarryVolek/PyTorch_Speaker_Verification/
            https://github.com/resemble-ai/Resemblyzer/

        Parameters
        ----------
        """
        # pre-processing and voice activity detection (VAD) part 1
        utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'],
                                      increase_only=True)
        utter = self.trim_long_silences(utter)

        # basically this does nothing if 60 is chosen, 60 is too high, so the whole wav will be selected.
        # This just makes an interval from beginning to the end.
        intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection part 2

        for interval_index, interval in enumerate(intervals):
            # if (interval[1] - interval[0]) > self.utter_min_len:  # If partial utterance is sufficiently long,
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
                                  win_length=int(
                                      self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                  hop_length=int(
                                      self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'],
                                            n_fft=self.params['preprocessing']['nfft'],
                                            n_mels=self.nmels)

            SS = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of partial utterance

        return SS

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



class anonymizer_loader:
    def __init__(self, cfg_path='./config/config.json', nmels=40):
        """
        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment
        """
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.file_path = self.params['file_path']
        self.nmels = nmels
        self.setup_cuda()
        self.main_df = pd.read_csv(os.path.join(self.params['file_path'], "PathologAnonym_project/all_70_30_contentmel.csv"), sep=';')
        # self.main_df = pd.read_csv(os.path.join(self.params['file_path'], "PathologAnonym_project/masterlist_org.csv"), sep=';')

        # self.main_df = self.main_df[self.main_df['subset'] == 'children']
        self.main_df = self.main_df[self.main_df['subset'] == 'adults']
        self.main_df = self.main_df[self.main_df['automatic_WRR'] > 0]
        self.main_df = self.main_df[self.main_df['age_y'] > 0]


        # criteria for choosing a subset of df




    def do_anonymize_nopitch(self):
        """
        """
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir")
        hifi_gan = hifi_gan.to(self.device)

        self.speaker_list = self.main_df['speaker_id'].unique().tolist()
        for speaker_name in tqdm(self.speaker_list):

            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]

            for index, row in selected_speaker_df.iterrows():

                row_relativepath = row['relative_path'].replace('.npy', '.wav')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysarthria_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysglossia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysphonia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/CLP_70_30_contentmel/PEAKS', 'PEAKS')

                original_path = os.path.join(self.file_path, row_relativepath)
                utterance, sr = sf.read(original_path)

                x_mel = self.get_mel_preproc(utterance)
                x_mel = x_mel.transpose(1,0)

                x_mel = torch.from_numpy(x_mel)
                x_mel = x_mel.float()
                x_mel = x_mel.unsqueeze(0)

                x_d2 = hifi_gan.decode_batch(x_mel.to(self.device)) # torch.Size([1, 80, 605])
                source_audio_reconstruced = x_d2.detach().cpu().numpy()[0,0]

                os.makedirs(os.path.dirname(original_path.replace('/PEAKS', '/PEAKS_onlyGAN_anonymized')), exist_ok=True)
                write(original_path.replace('/PEAKS', '/PEAKS_onlyGAN_anonymized'), 16000, source_audio_reconstruced)



    def do_anonymize(self):
        """
        """
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-libritts-16kHz", savedir="tmpdir")
        hifi_gan = hifi_gan.to(self.device)

        self.speaker_list = self.main_df['speaker_id'].unique().tolist()
        for speaker_name in tqdm(self.speaker_list):
            mean_noise = random.uniform(0, 0.2)
            std_noise = random.uniform(0, 0.005)

            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]

            age_speaker = selected_speaker_df['age_y'].values[0]
            gender_speaker = selected_speaker_df['gender'].values[0]
            pitch_shift = random.uniform(-1.0, -0.7)

            if gender_speaker == 'female':
                pitch_shift = random.uniform(-1.2, -0.8)
                if age_speaker < 8 and age_speaker > 0:
                    pitch_shift = random.uniform(-1.2, -1.0)

            if gender_speaker == 'male':
                randdd = random.randint(0,1)
                if randdd > 0:
                    pitch_shift = random.uniform(-1.2, -0.8)
                else:
                    pitch_shift = random.uniform(0.8, 1.0)
                    if age_speaker < 10 and age_speaker > 0:
                        pitch_shift = random.uniform(0.5, 0.8)
                    elif age_speaker > 10 and age_speaker < 20:
                        pitch_shift = random.uniform(0.6, 1.0)
                    elif age_speaker > 20:
                        pitch_shift = random.uniform(0.8, 1.2)

            for index, row in selected_speaker_df.iterrows():
                row_relativepath = row['relative_path'].replace('.npy', '.wav')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysarthria_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysglossia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysphonia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/CLP_70_30_contentmel/PEAKS', 'PEAKS')

                original_path = os.path.join(self.file_path, row_relativepath)

                # original_path = os.path.join(self.file_path, row['relative_path'])
                utterance, sr = sf.read(original_path)

                x_mel = self.get_mel_preproc(utterance)
                x_d2 = librosa.effects.pitch_shift(x_mel, sr=16000, n_steps=pitch_shift, bins_per_octave=12)
                x_d2 += np.random.normal(mean_noise, std_noise, x_d2.shape)

                x_d2 = torch.from_numpy(x_d2)
                x_d2 = x_d2.float()
                x_d2 = x_d2.T[None]
                x_d2 = x_d2.to(self.device)
                x_d2 = hifi_gan.decode_batch(x_d2)
                source_audio_pitched = x_d2.detach().cpu().numpy()[0,0]

                # if pitch shift positive, denoise
                if pitch_shift > 0:
                    source_audio_pitched = nr.reduce_noise(y=source_audio_pitched, sr=16000)

                os.makedirs(os.path.dirname(original_path.replace('/PEAKS', '/PEAKS_Pitch_anonymized')), exist_ok=True)
                write(original_path.replace('/PEAKS', '/PEAKS_Pitch_anonymized'), 16000, source_audio_pitched)

    def get_mel_preproc(self, x):
        """
        References:
            https://github.com/RF5/simple-autovc/
            https://github.com/CODEJIN/AutoVC/

        Parameters
        ----------
        """
        mel_basis_hifi = mel(self.params['preprocessing']['sr'], 1024, fmin=0, fmax=8000, n_mels=80).T
        b, a = self.butter_highpass(30, self.params['preprocessing']['sr'], order=5)

        # Remove drifting noise
        wav = signal.filtfilt(b, a, x)

        # Ddd a little random noise for model roubstness
        wav = wav * 0.96 + (np.random.RandomState().rand(wav.shape[0]) - 0.5) * 1e-06

        # Compute spect
        D = self.pySTFT(wav).T
        # Convert to mel and normalize
        D_mel = np.dot(D, mel_basis_hifi)
        mel_spec = np.log(np.clip(D_mel, 1e-5, float('inf'))).astype(np.float32)

        return mel_spec


    def provide_data_anonymized(self):
        """
        Returns
        -------
        speakers: dictionary of list
            a dictionary of all the speakers. Each speaker contains a list of
            all its utterances converted to mel spectrograms
        """
        # dictionary of speakers
        speakers = {}

        for speaker_name in tqdm(self.speaker_list):
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]
            # list of utterances of each speaker
            utterances = []

            for index, row in selected_speaker_df.iterrows():
                path = os.path.join(self.file_path, row['relative_path'])
                path_anonymized = path.replace('/PEAKS', '/PEAKS_Pitch_anonymized')
                utter, sr = sf.read(path_anonymized)
                utterance = self.tisv_preproc(utter)
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances

        return speakers


    def tisv_preproc(self, utter):
        """
        GE2E-loss-based pre-processing
        References:
            https://github.com/JanhHyun/Speaker_Verification/
            https://github.com/HarryVolek/PyTorch_Speaker_Verification/
            https://github.com/resemble-ai/Resemblyzer/

        Parameters
        ----------
        """
        # pre-processing and voice activity detection (VAD) part 1
        utter = self.normalize_volume(utter, self.params['preprocessing']['audio_norm_target_dBFS'],
                                      increase_only=True)
        utter = self.trim_long_silences(utter)

        # basically this does nothing if 60 is chosen, 60 is too high, so the whole wav will be selected.
        # This just makes an interval from beginning to the end.
        intervals = librosa.effects.split(utter, top_db=30)  # voice activity detection part 2

        for interval_index, interval in enumerate(intervals):
            # if (interval[1] - interval[0]) > self.utter_min_len:  # If partial utterance is sufficiently long,
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
                                  win_length=int(
                                      self.params['preprocessing']['window'] * self.params['preprocessing']['sr']),
                                  hop_length=int(
                                      self.params['preprocessing']['hop'] * self.params['preprocessing']['sr']))
            S = np.abs(S) ** 2
            mel_basis = librosa.filters.mel(sr=self.params['preprocessing']['sr'],
                                            n_fft=self.params['preprocessing']['nfft'],
                                            n_mels=self.nmels)

            SS = np.log10(np.dot(mel_basis, S) + 1e-6)  # log mel spectrogram of partial utterance

        return SS


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
            torch.cuda.manual_seed_all(self.model_info['seed'])
            torch.manual_seed(self.model_info['seed'])
        else:
            self.device = torch.device('cpu')




class original_dvector_loader:
    def __init__(self, cfg_path='./configs/config.json', M=8):
        """For thresholding and testing.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        M: int
            number of utterances per speaker
            must be an even number

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
            return shape: (# all speakers, M, embedding size)
        """
        params = read_config(cfg_path)
        self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original'], "*.npy"))
        self.M = M


    def provide_test_original(self):
        output_tensor = []

        # return all speakers
        for speaker in self.speaker_list:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            else:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor.append(embedding)
        output_tensor = np.stack(output_tensor)
        output_tensor = torch.from_numpy(output_tensor)

        return output_tensor



class anonymized_dvector_loader:
    def __init__(self, cfg_path='./configs/config.json', M=8):
        """For d-vector calculation.

        Parameters
        ----------
        cfg_path: str
            Config file path of the experiment

        M: int
            number utterances per speaker
            must be an even number

        Returns
        -------
        output_tensor: torch tensor
            loaded mel spectrograms
            return shape: (# all speakers, M, embedding size)
        """
        params = read_config(cfg_path)
        self.speaker_list_anonymized = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_anonymized'], "*.npy"))
        self.speaker_list_original = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original'], "*.npy"))
        self.M = M
        self.speaker_list_anonymized.sort()
        self.speaker_list_original.sort()
        assert len(self.speaker_list_original) == len(self.speaker_list_anonymized)


    def provide_test_anonymized_and_original(self):
        output_tensor_anonymized = []
        output_tensor_original = []

        # return all speakers of anonymized
        for speaker in self.speaker_list_anonymized:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            else:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor_anonymized.append(embedding)
        output_tensor_anonymized = np.stack(output_tensor_anonymized)
        output_tensor_anonymized = torch.from_numpy(output_tensor_anonymized)


        # return all speakers of original
        for speaker in self.speaker_list_original:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            else:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor_original.append(embedding)
        output_tensor_original = np.stack(output_tensor_original)
        output_tensor_original = torch.from_numpy(output_tensor_original)

        return output_tensor_anonymized, output_tensor_original
