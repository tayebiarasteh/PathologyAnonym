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
import scipy

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
        self.main_df = pd.read_csv(os.path.join(self.params['file_path'], "PathologAnonym_project/all_70_30_contentmel.csv"), sep=';')


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

                row_relativepath = row['relative_path'].replace('.npy', '.wav')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysarthria_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysglossia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysphonia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/CLP_70_30_contentmel/PEAKS', 'PEAKS')


                utter, sr = sf.read(os.path.join(self.file_path, row_relativepath))
                utterance = self.tisv_preproc(utter)
                utterance = torch.from_numpy(np.transpose(utterance, axes=(1, 0)))
                utterances.append(utterance)
            speakers[speaker_name] = utterances

        return speakers


    def provide_data_anonymized(self, anonym_utter_dirname='05'):
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

                row_relativepath = row['relative_path'].replace('.npy', '.wav')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysarthria_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysglossia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysphonia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/CLP_70_30_contentmel/PEAKS', 'PEAKS')

                path = os.path.join(self.file_path, row_relativepath)
                path_anonymized = path.replace('/PEAKS', '/PEAKS_' + anonym_utter_dirname + '_mcadams_anonymized')
                # path_anonymized = path.replace('/PEAKS', '/PEAKS_onlyGAN_anonymized')
                # path_anonymized = path.replace('/PEAKS', '/PEAKS_random_mcadams_anonymized')
                # path_anonymized = path.replace('/PEAKS', '/PEAKS_anonymized')
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

        self.main_df = self.main_df[self.main_df['subset'] == 'children']
        # self.main_df = self.main_df[self.main_df['subset'] == 'adults']
        self.main_df = self.main_df[self.main_df['automatic_WRR'] > 0]
        self.main_df = self.main_df[self.main_df['age_y'] > 0]



    def single_anonymize(self, utterance, sr, output_path, winLengthinms=20, shiftLengthinms=10, lp_order=20, mcadams=0.8):
        """
        This mcadams anonymizer script is taken directly from
        https://github.com/Voice-Privacy-Challenge/Voice-Privacy-Challenge-2022/blob/master/baseline/local/anon/anonymise_dir_mcadams.py

        @author: Jose Patino, Massimiliano Todisco, Pramod Bachhav, Nicholas Evans
        Audio Security and Privacy Group, EURECOM
        https://github.com/zhu00121/Anonymized-speech-diagnostics/blob/main/Local/Anonymization/McAdams/mcadams.py

        modified by: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
        https://github.com/tayebiarasteh/
        """
        eps = np.finfo(np.float32).eps
        utterance = utterance + eps

        # simulation parameters
        winlen = np.floor(winLengthinms * 0.001 * sr).astype(int)
        shift = np.floor(shiftLengthinms * 0.001 * sr).astype(int)
        length_sig = len(utterance)

        # fft processing parameters
        NFFT = 2 ** (np.ceil((np.log2(winlen)))).astype(int)
        # anaysis and synth window which satisfies the constraint
        wPR = np.hanning(winlen)
        K = np.sum(wPR) / shift
        win = np.sqrt(wPR / K)
        Nframes = 1 + np.floor((length_sig - winlen) / shift).astype(int)  # nr of complete frames

        # carry out the overlap - add FFT processing
        sig_rec = np.zeros([length_sig])  # allocate output+'ringing' vector

        for m in np.arange(1, Nframes):
            # indices of the mth frame
            index = np.arange(m * shift, np.minimum(m * shift + winlen, length_sig))
            # windowed mth frame (other than rectangular window)
            frame = utterance[index] * win
            # get lpc coefficients
            a_lpc = librosa.core.lpc(frame + eps, lp_order)
            # get poles
            poles = scipy.signal.tf2zpk(np.array([1]), a_lpc)[1]
            # index of imaginary poles
            ind_imag = np.where(np.isreal(poles) == False)[0]
            # index of first imaginary poles
            ind_imag_con = ind_imag[np.arange(0, np.size(ind_imag), 2)]

            # here we define the new angles of the poles, shifted accordingly to the mcadams coefficient
            # values >1 expand the spectrum, while values <1 constract it for angles>1
            # values >1 constract the spectrum, while values <1 expand it for angles<1
            # the choice of this value is strongly linked to the number of lpc coefficients
            # a bigger lpc coefficients number constraints the effect of the coefficient to very small variations
            # a smaller lpc coefficients number allows for a bigger flexibility
            new_angles = np.angle(poles[ind_imag_con]) ** mcadams

            # make sure new angles stay between 0 and pi
            new_angles[np.where(new_angles >= np.pi)] = np.pi
            new_angles[np.where(new_angles <= 0)] = 0

            # copy of the original poles to be adjusted with the new angles
            new_poles = poles
            for k in np.arange(np.size(ind_imag_con)):
                # compute new poles with the same magnitued and new angles
                new_poles[ind_imag_con[k]] = np.abs(poles[ind_imag_con[k]]) * np.exp(1j * new_angles[k])
                # applied also to the conjugate pole
                new_poles[ind_imag_con[k] + 1] = np.abs(poles[ind_imag_con[k] + 1]) * np.exp(-1j * new_angles[k])

                # recover new, modified lpc coefficients
            a_lpc_new = np.real(np.poly(new_poles))
            # get residual excitation for reconstruction
            res = scipy.signal.lfilter(a_lpc, np.array(1), frame)
            # reconstruct frames with new lpc coefficient
            frame_rec = scipy.signal.lfilter(np.array([1]), a_lpc_new, res)
            frame_rec = frame_rec * win

            outindex = np.arange(m * shift, m * shift + len(frame_rec))
            # overlap add
            sig_rec[outindex] = sig_rec[outindex] + frame_rec
        sig_rec = sig_rec / np.max(np.abs(sig_rec))

        scipy.io.wavfile.write(output_path, sr, np.float32(sig_rec))


    def do_anonymize(self, mcadams_coef=0.8, output_utter_dirname='PEAKS_random05_mcadams_anonymized'):
        self.speaker_list = self.main_df['speaker_id'].unique().tolist()
        for speaker_name in tqdm(self.speaker_list):

            # mcadams_coef = random.uniform(0.75, 0.9)
            selected_speaker_df = self.main_df[self.main_df['speaker_id'] == speaker_name]

            for index, row in selected_speaker_df.iterrows():
                row_relativepath = row['relative_path'].replace('.npy', '.wav')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysarthria_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysglossia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/dysphonia_70_30_contentmel/PEAKS', 'PEAKS')
                row_relativepath = row_relativepath.replace('tisv_preprocess/CLP_70_30_contentmel/PEAKS', 'PEAKS')
                original_path = os.path.join(self.file_path, row_relativepath)
                # original_path = os.path.join(self.file_path, row['relative_path'])

                utterance, sr = sf.read(original_path)
                os.makedirs(os.path.dirname(original_path.replace('/PEAKS', '/' + output_utter_dirname)), exist_ok=True)
                output_path = original_path.replace('/PEAKS', '/' + output_utter_dirname)

                self.single_anonymize(utterance=utterance, sr=sr, output_path=output_path, mcadams=mcadams_coef)



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
    def __init__(self, cfg_path='./configs/config.json', M=8, subsetname='dysphonia'):
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
        if subsetname == 'dysphonia':
            self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_dysphonia'], "*.npy"))
        elif subsetname == 'dysarthria':
            self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_dysarthria'], "*.npy"))
        elif subsetname == 'dysglossia':
            self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_dysglossia'], "*.npy"))
        elif subsetname == 'CLP':
            self.speaker_list = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_CLP'], "*.npy"))

        self.M = M


    def provide_test_original(self):
        output_tensor = []

        # return all speakers
        for speaker in self.speaker_list:
            embedding = np.load(speaker)

            # randomly sample a fixed specified length
            if embedding.shape[0] == self.M:
                id = np.array([0])
            elif embedding.shape[0] > self.M:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            else:
                diff = self.M - embedding.shape[0]
                id = np.array([0])
                embedding = np.vstack((embedding, embedding[0:diff]))

            embedding = embedding[id[0]:id[0] + self.M]
            output_tensor.append(embedding)
        output_tensor = np.stack(output_tensor)
        output_tensor = torch.from_numpy(output_tensor)

        return output_tensor



class anonymized_dvector_loader:
    def __init__(self, cfg_path='./configs/config.json', M=8, subsetname='dysphonia'):
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
        if subsetname == 'dysphonia':
            self.speaker_list_original = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_dysphonia'], "*.npy"))
            self.speaker_list_anonymized = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_anony_dysphonia'], "*.npy"))

        elif subsetname == 'dysarthria':
            self.speaker_list_original = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_dysarthria'], "*.npy"))
            self.speaker_list_anonymized = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_anony_dysarthria'], "*.npy"))

        elif subsetname == 'dysglossia':
            self.speaker_list_original = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_dysglossia'], "*.npy"))
            self.speaker_list_anonymized = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_anony_dysglossia'], "*.npy"))

        elif subsetname == 'CLP':
            self.speaker_list_original = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_original_CLP'], "*.npy"))
            self.speaker_list_anonymized = glob.glob(os.path.join(params['target_dir'], params['dvectors_path_anony_CLP'], "*.npy"))

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
            elif embedding.shape[0] > self.M:
                id = np.random.randint(0, embedding.shape[0] - self.M, 1)
            else:
                diff = self.M - embedding.shape[0]
                id = np.array([0])
                embedding = np.vstack((embedding, embedding[0:diff]))

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
