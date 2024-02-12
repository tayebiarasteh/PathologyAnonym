"""
graphs.py
Created on Feb 12, 2024.

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@fau.de>
https://github.com/tayebiarasteh/
"""
import pdb
import os
import matplotlib.pyplot as plt
import numpy as np



import warnings
warnings.filterwarnings('ignore')





def auroc():
    # Data
    x = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # dysarthria
    y = np.array([97.05, 96.05, 94.06, 96.29, 98.27, 98.48])
    yerr = np.array([0.39, 0.65, 0.92, 0.60, 0.28, 0.39])
    ref_auroc = 97.33

    # dysglossia
    # y = np.array([98.62, 97.69, 98.68, 97.55, 98.12, 98.52])
    # yerr = np.array([0.25, 0.44, 0.23, 0.28, 0.31, 0.24])
    # ref_auroc = 97.73

    # dysphonia
    # y = np.array([96.60, 98.02, 97.62, 97.71, 98.63, 96.19])
    # yerr = np.array([0.64, 0.51, 0.47, 0.57, 0.35, 0.59])
    # ref_auroc = 99.12

    # Creating the plot
    plt.figure(figsize=(10, 8))

    # Plotting the data points with error bars and thicker lines
    plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='black', capsize=5, linestyle='-', color='navy', linewidth=2,
                 label='McAdams Coefficient Anonym', markersize=8)

    # Reference standard AUROC value with a thicker dotted line
    plt.axhline(y=ref_auroc, color='green', linestyle='--', linewidth=2, label='Original Speech')

    # plt.title('Dysglossia Classification', fontsize=20)
    # plt.title('Dysarthria Classification', fontsize=18)

    plt.xlabel('McAdams Coefficients', fontsize=18)
    plt.ylabel('AUROC [%]', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Setting the Y-axis range
    plt.ylim([93, 100])
    # Annotating data points with their Y values
    for i, txt in enumerate(y):
        plt.annotate(f'{txt:.2f}', (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14)

    # Annotating the reference line value
    plt.text(0.8, ref_auroc, f'Original Speech: {ref_auroc:.2f}', va='center', ha='right', backgroundcolor='w', fontsize=14)

    # Adding legend with updated formatting
    plt.legend(fontsize=16)

    # Saving the figure
    # plt.savefig('final_thicker_lines_auroc_comparison.png', dpi=300)
    plt.show()


def eer():
    # Data
    x = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    # dysarthria
    y = np.array([51.37 , 49.87, 43.71, 36.63, 31.96, 16.22 ])
    yerr = np.array([0.28, 0.31, 0.64, 0.48, 0.45, 0.42])
    ref_auroc = 1.80

    # dysglossia
    y = np.array([50.94, 45.39, 41.53 , 36.79, 26.37, 9.35])
    yerr = np.array([0.42, 0.53, 0.69, 0.75, 0.72, 0.57])
    ref_auroc = 1.78
    #
    # # dysphonia
    y = np.array([50.46 , 48.50, 43.55 , 40.33, 35.93 , 12.35])
    yerr = np.array([0.55, 0.25, 0.63, 0.41, 0.52, 0.57])
    ref_auroc = 2.19

    # Creating the plot
    plt.figure(figsize=(10, 8))

    # Plotting the data points with error bars and thicker lines
    plt.errorbar(x, y, yerr=yerr, fmt='o', ecolor='black', capsize=5, linestyle='-', color='red', linewidth=2,
                 label='McAdams Coefficient Anonym', markersize=8)

    # Reference standard AUROC value with a thicker dotted line
    plt.axhline(y=ref_auroc, color='darkorange', linestyle='--', linewidth=2, label='Original Speech')

    # plt.title('Dysglossia Classification', fontsize=20)
    # plt.title('Dysarthria Classification', fontsize=18)

    plt.xlabel('McAdams Coefficients', fontsize=18)
    plt.ylabel('EER [%]', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    # Setting the Y-axis range
    plt.ylim([0, 55])
    # Annotating data points with their Y values
    for i, txt in enumerate(y):
        plt.annotate(f'{txt:.2f}', (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=14)

    # Annotating the reference line value
    plt.text(0.95, ref_auroc, f'Original Speech: {ref_auroc:.2f}', va='center', ha='right', backgroundcolor='w', fontsize=14)

    # Adding legend with updated formatting
    plt.legend(fontsize=16)

    # Saving the figure
    # plt.savefig('final_thicker_lines_auroc_comparison.png', dpi=300)
    plt.show()



if __name__ == '__main__':
    auroc()
    # eer()