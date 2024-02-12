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





def mcadams_anonymization_process_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
                                      mcadams_coef=0.8, output_utter_dirname='PEAKS_random05_mcadams_anonymized'):
    """

    Parameters
    ----------
    global_config_path: str
        always global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml"
    """
    print('\nanonymizing all utteraces of each speaker in the same way....')
    print('loop over speakers....')
    data_handler_anonymizer = anonymizer_loader(cfg_path=global_config_path, nmels=40)

    data_handler_anonymizer.do_anonymize(mcadams_coef=mcadams_coef, output_utter_dirname=output_utter_dirname)
    print('anonymization done!')



def anonymized_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
        experiment_name='baseline_speaker_model', epochs=1000, M=8, spk_nmels=40, anonym_utter_dirname='05', subsetname='dysphonia', subset='children'):
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

    data_handler.main_df = data_handler.main_df[data_handler.main_df['subset'] == subset]

    if subsetname == 'dysphonia':
        data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'logitech']

    if subsetname == 'dysarthria':
        data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'plantronics']

    if subsetname == 'dysglossia':
        data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'maxillofacial']

    data_handler.main_df = data_handler.main_df[data_handler.main_df['patient_control'] == 'patient']


    data_loader = data_handler.provide_data_anonymized(anonym_utter_dirname=anonym_utter_dirname)
    print('\npreprocessing done!')

    print('Creating the d-vectors (network prediction) for the anonymized signals....')
    predictor.dvector_prediction(data_loader, anonymized=True, subsetname=subsetname)

    print('\nEER calculation....')
    avg_EER_test, std_EER, numspk = predictor.EER_newmethod_epochy_anonymized(cfg_path, M=M, epochs=epochs, subsetname=subsetname)

    print('\n------------------------------------------------------'
          '----------------------------------')
    print(f'Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n '
          f'No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}')
    print(f'\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%')

    # saving the stats
    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}" \
           f"\n\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_anonymized_M' + str(M) + subsetname + anonym_utter_dirname, 'a') as f:
        f.write(mesg)




def direct_clssical_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
                   experiment_name='baseline_speaker_model', epochs=1000, M=8, spk_nmels=40, subsetname='dysphonia', subset='children'):
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

    data_handler.main_df = data_handler.main_df[data_handler.main_df['subset'] == subset]

    if subsetname == 'dysphonia':
        data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'logitech']

    if subsetname == 'dysarthria':
        data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'plantronics']

    if subsetname == 'dysglossia':
        data_handler.main_df = data_handler.main_df[data_handler.main_df['mic_room'] == 'maxillofacial']


    data_handler.main_df = data_handler.main_df[data_handler.main_df['patient_control'] == 'patient']

    data_loader = data_handler.provide_data_original()
    print('\npreprocessing done!')

    print('Creating the d-vectors (network prediction)....')
    predictor.dvector_prediction(data_loader, anonymized=False, subsetname=subsetname)

    print('\nEER calculation....')
    avg_EER_test, std_EER, numspk = predictor.EER_newmethod_epochy(cfg_path, M=M, epochs=epochs, subsetname=subsetname)

    print('\n------------------------------------------------------'
          '----------------------------------')
    print(f'Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n '
          f'No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}')
    print(f'\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%')

    # saving the stats
    mesg = f'\n----------------------------------------------------------------------------------------\n' \
           f"Speaker model: GE2E trained on Librispeech | speaker model No. mels: {int(spk_nmels)}\n No. enrolment/verification utterances per speaker: {int(M/2)}/{int(M/2)} | No. speakers: {int(numspk)}" \
           f"\n\n\tAverage EER over {epochs} repetitions: {(avg_EER_test) * 100:.2f} ± {(std_EER) * 100:.2f}%" \
           f'\n\n----------------------------------------------------------------------------------------\n'
    with open(os.path.join(params['target_dir'], params['stat_log_path']) + '/test_results_original_M' + str(M) + subsetname, 'a') as f:
        f.write(mesg)





if __name__ == '__main__':
    # mcadams_coef = random.uniform(0.5, 0.9)
    mcadams_coef = 0.5
    # mcadams_coef = 0.6
    # mcadams_coef = 0.7
    # mcadams_coef = 0.8
    # mcadams_coef = 0.9
    # mcadams_coef = 1.0
    # mcadams_coef = 1.1

    if np.abs(0.5 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_05_mcadams_anonymized'
    elif np.abs(0.6 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_06_mcadams_anonymized'
    elif np.abs(0.7 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_07_mcadams_anonymized'
    elif np.abs(0.8 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_08_mcadams_anonymized'
    elif np.abs(0.9 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_09_mcadams_anonymized'
    elif np.abs(1.0 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_10_mcadams_anonymized'
    elif np.abs(1.1 - mcadams_coef) < 0.0001:
        output_utter_dirname = 'PEAKS_11_mcadams_anonymized'

    print(output_utter_dirname)

    # mcadams_anonymization_process_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
    #                                   mcadams_coef=mcadams_coef, output_utter_dirname=output_utter_dirname)
    #
    # direct_clssical_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
    #                    experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, subsetname='dysphonia')
    # anonymized_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
    #                    experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, anonym_utter_dirname=output_utter_dirname, subsetname='dysphonia')



    # dysarthria
    direct_clssical_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
                       experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, subsetname='dysarthria')
    for idx in ['05', '06', '07', '08', '09', '10', '11']:
        anonymized_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
            experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, anonym_utter_dirname=idx,
            subsetname='dysarthria', subset='adults')

    ##############################################################################


    # dysglossia
    direct_clssical_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
                       experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, subsetname='dysglossia')
    for idx in ['05', '06', '07', '08', '09', '10', '11']:
        anonymized_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
            experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, anonym_utter_dirname=idx,
            subsetname='dysglossia', subset='adults')

    ##############################################################################


    # dysphonia
    direct_clssical_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
                       experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, subsetname='dysphonia')
    for idx in ['05', '06', '07', '08', '09', '10', '11']:
        anonymized_EER_calculation_e2e(global_config_path="/home/soroosh/Documents/Repositories/PathologyAnonym/mcAdams_Anonym/config/config.yaml",
            experiment_name='baseline_speaker_model', epochs=10, M=8, spk_nmels=40, anonym_utter_dirname=idx,
            subsetname='dysphonia', subset='adults')

    ##############################################################################


