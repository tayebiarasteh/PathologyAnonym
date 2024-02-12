"""
inference_speaker.py
Created on Oct 30, 2023.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pdb
import os
import numpy as np

from config.serde import open_experiment
from inference_speaker_data_loader import loader_for_dvector_creation, anonymizer_loader
from speaker_Prediction import Prediction
from models.lstm import SpeechEmbedder


import warnings
warnings.filterwarnings('ignore')





def anonymization_process_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml"):
    """

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml"
    """
    print('\nanonymizing all utteraces of each speaker in the same way....')
    print('loop over speakers....')
    data_handler_anonymizer = anonymizer_loader(cfg_path=global_config_path, nmels=40)


    data_handler_anonymizer.do_anonymize_nopitch()
    # data_handler_anonymizer.do_anonymize()
    print('anonymization done!')



def anonymized_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml",
        experiment_name='baseline_speaker_model', epochs=1000, M=8, spk_nmels=40):
    """

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml"

    epochs: int
        total number of epochs to do the evaluation process.
        The results will be the average over the result of
        each epoch.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']

    # d-vector and EER calculation
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    predictor.setup_model_for_inference(model=model)
    # d-vector creation
    print('preprocessing for d-vector creation....')
    data_handler = loader_for_dvector_creation(cfg_path=cfg_path, spk_nmels=spk_nmels)


    # criteria for choosing a subset of df
    data_handler.main_df = data_handler.main_df[data_handler.main_df['subset'] == 'children']
    data_handler.main_df = data_handler.main_df[data_handler.main_df['patient_control'] == 'patient']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'dnt']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'Plantronics']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'Logitech']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'Mixed']

    data_loader = data_handler.provide_data_anonymized()
    print('preprocessing done!')

    print('Creating the d-vectors (network prediction) for the anonymized signals....')
    predictor.dvector_prediction(data_loader, anonymized=True)

    print('EER calculation....')
    avg_EER_test, std_EER, numspk = predictor.EER_newmethod_epochy_anonymized(cfg_path, M=M, epochs=epochs)

    print('\n------------------------------------------------------'
          '----------------------------------')
    print(f'Speaker model: GE2E trained on 3K PEAKS speakers combined | speaker model No. mels: {int(spk_nmels)}\n '
          f'No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}')
    print(f'\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%')

    # saving the stats
    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: GE2E trained on 3K PEAKS speakers combined | speaker model No. mels: {int(spk_nmels)}\n No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}" \
           f"\n\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_anonymized_M' + str(M), 'a') as f:
        f.write(mesg)




def direct_clssical_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml",
                   experiment_name='baseline_speaker_model', epochs=1000, M=8, spk_nmels=40):
    """Main function for creating d-vectors & testing, for different models based on epochs
    Purpose here is validation of the model

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml"

    experiment_name: str
        name of the experiment, in case of resuming training.
        name of new experiment, in case of new training.

    epochs: int
        total number of epochs to do the evaluation process.
        The results will be the average over the result of
        each epoch.
    """
    params = open_experiment(experiment_name, global_config_path)
    cfg_path = params['cfg_path']
    predictor = Prediction(cfg_path)
    model = SpeechEmbedder(nmels=params['preprocessing']['nmels'], hidden_dim=params['Network']['hidden_dim'],
                           output_dim=params['Network']['output_dim'], num_layers=params['Network']['num_layers'])

    predictor.setup_model_for_inference(model=model)

    # d-vector creation
    print('preprocessing for d-vector creation....')
    data_handler = loader_for_dvector_creation(cfg_path=cfg_path, spk_nmels=spk_nmels)

    # criteria for choosing a subset of df
    data_handler.main_df = data_handler.main_df[data_handler.main_df['subset'] == 'children']
    data_handler.main_df = data_handler.main_df[data_handler.main_df['patient_control'] == 'patient']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'dnt']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'Plantronics']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'Logitech']
    # data_handler.main_df = data_handler.main_df[data_handler.main_df['microphone'] == 'Mixed']

    data_loader = data_handler.provide_data_original()
    print('preprocessing done!')

    print('Creating the d-vectors (network prediction)....')
    predictor.dvector_prediction(data_loader, anonymized=False)

    print('EER calculation....')
    avg_EER_test, std_EER, numspk = predictor.EER_newmethod_epochy(cfg_path, M=M, epochs=epochs)

    print('\n------------------------------------------------------'
          '----------------------------------')
    print(f'Speaker model: GE2E trained on 3K PEAKS speakers combined | speaker model No. mels: {int(spk_nmels)}\n '
          f'No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}')
    print(f'\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%')

    # saving the stats
    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: GE2E trained on 3K PEAKS speakers combined | speaker model No. mels: {int(spk_nmels)}\n No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}" \
           f"\n\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_original_M' + str(M), 'a') as f:
        f.write(mesg)





if __name__ == '__main__':

    anonymization_process_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml")

    # direct_clssical_EER_calculation_e2e(global_config_path="/home/arasteh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml",
    #                    experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40)
    # anonymized_EER_calculation_e2e(global_config_path="/home/arasteh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml",
    #                    experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40)

    # direct_clssical_EER_calculation_e2e(global_config_path="/home/arasteh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml",
    #                    experiment_name='baseline_speaker_model_librispeech', epochs=10, M=8, spk_nmels=40)
    # anonymized_EER_calculation_e2e(global_config_path="/home/arasteh/Documents/Repositories/PathologyAnonym/Pitch_Anonym/config/config.yaml",
    #                    experiment_name='baseline_speaker_model_librispeech', epochs=10, M=8, spk_nmels=40)
