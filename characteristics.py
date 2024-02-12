"""
characteristics.py
Created on Feb 12, 2024.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pandas as pd




import warnings
warnings.filterwarnings('ignore')




# Load the dataset
data_path = '/home/soroosh/Documents/datasets/anonymization/PathologAnonym_project/masterlist_org.csv'
adults_data = pd.read_csv(data_path, delimiter=';')

# Filter dataset for adults only
adults_data = adults_data[adults_data['subset'] == 'adults']
# adults_data = adults_data[adults_data['subset'] == 'children']
adults_data = adults_data[adults_data['patient_control'] == 'patient']
adults_data = adults_data[adults_data['mic_room'] == 'plantronics']

# Total number of speakers
total_speakers_adults = adults_data['speaker_id'].nunique()
female_speakers_adults = adults_data[adults_data['gender'] == 'female']['speaker_id'].nunique()
male_speakers_adults = adults_data[adults_data['gender'] == 'male']['speaker_id'].nunique()

# Total number of utterances
total_utterances_adults = len(adults_data)
female_utterances_adults = len(adults_data[adults_data['gender'] == 'female'])
male_utterances_adults = len(adults_data[adults_data['gender'] == 'male'])

# Total duration in hours
total_duration_hours_adults = adults_data['file_length'].sum() / 3600
female_duration_hours_adults = adults_data[adults_data['gender'] == 'female']['file_length'].sum() / 3600
male_duration_hours_adults = adults_data[adults_data['gender'] == 'male']['file_length'].sum() / 3600

# Calculate percentages
female_percentage_adults = (female_speakers_adults / total_speakers_adults) * 100
male_percentage_adults = (male_speakers_adults / total_speakers_adults) * 100

# Average age and WRR per speaker
age_wrr_per_speaker_adults = adults_data.groupby(['speaker_id', 'gender']).agg({'age_y': 'mean', 'automatic_WRR': 'mean'}).reset_index()

# Calculate mean and std for age and WRR
female_age_mean_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'female']['age_y'].mean()
female_age_std_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'female']['age_y'].std()
male_age_mean_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'male']['age_y'].mean()
male_age_std_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'male']['age_y'].std()

female_wrr_mean_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'female']['automatic_WRR'].mean()
female_wrr_std_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'female']['automatic_WRR'].std()
male_wrr_mean_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'male']['automatic_WRR'].mean()
male_wrr_std_adults = age_wrr_per_speaker_adults[age_wrr_per_speaker_adults['gender'] == 'male']['automatic_WRR'].std()

# Calculate overall mean and std for age and WRR for adults
overall_age_mean_adults = age_wrr_per_speaker_adults['age_y'].mean()
overall_age_std_adults = age_wrr_per_speaker_adults['age_y'].std()

overall_wrr_mean_adults = age_wrr_per_speaker_adults['automatic_WRR'].mean()
overall_wrr_std_adults = age_wrr_per_speaker_adults['automatic_WRR'].std()


# Display results
results_adults = {
    "Total number of speakers": f"{total_speakers_adults} (Overall) / {female_percentage_adults:.0f}% (Female) / {male_percentage_adults:.0f}% (Male)",
    "Total number of utterances": f"{total_utterances_adults} (Overall) / {female_utterances_adults / total_utterances_adults * 100:.0f}% (Female) / {male_utterances_adults / total_utterances_adults * 100:.0f}% (Male)",
    "Total duration": f"{total_duration_hours_adults:.2f} hours (Overall) / {female_duration_hours_adults / total_duration_hours_adults * 100:.0f}% (Female) / {male_duration_hours_adults / total_duration_hours_adults * 100:.0f}% (Male)",
    " age": f"Overall: {overall_age_mean_adults:.2f} ± {overall_age_std_adults:.2f} [{female_age_mean_adults:.2f} ± {female_age_std_adults:.2f} (Female) / {male_age_mean_adults:.2f} ± {male_age_std_adults:.2f} (Male)",
    " WRR": f"Overall: {overall_wrr_mean_adults:.2f} ± {overall_wrr_std_adults:.2f} [{female_wrr_mean_adults:.2f} ± {female_wrr_std_adults:.2f} (Female) / {male_wrr_mean_adults:.2f} ± {male_wrr_std_adults:.2f} (Male)]"
}

for key, value in results_adults.items():
    print(f"{key}: {value}")
