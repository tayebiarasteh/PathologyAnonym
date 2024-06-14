# The Impact of Speech Anonymization on Pathology and Its Limits



Overview
------

* This is the official repository of the paper [**The Impact of Speech Anonymization on Pathology and Its Limits**](https://arxiv.org/abs/2404.08064).

Abstract
------
....

### Prerequisites

The software is developed in **Python 3.9**. For the deep learning, the **PyTorch 1.13** framework is used.



Main Python modules required for the software can be installed from ./requirements:

```
$ conda env create -f requirements.yaml
$ conda activate pathology_anonym
```

**Note:** This might take a few minutes.


Code structure
---

Our source code for training and evaluation of the deep neural networks, speech analysis and preprocessing are available here.

1. Everything can be run from *./PathologyAnonym_main.py*. 
* The data preprocessing parameters, directories, hyper-parameters, and model parameters can be modified from *./configs/config.yaml*.
* Also, you should first choose an `experiment` name (if you are starting a new experiment) for training, in which all the evaluation and loss value statistics, tensorboard events, and model & checkpoints will be stored. Furthermore, a `config.yaml` file will be created for each experiment storing all the information needed.
* For testing, just load the experiment which its model you need.

2. The rest of the files:
* *./data/* directory contains all the data preprocessing, and loading files.
* *./mcAdams_Anonym/* directory contains all the files for anonymization using McAdams coefficient method.
* *./Pitch_Anonym/* directory contains all the files for anonymization using training-based randomized pitch shift + HiFi-GAN method.
* *./PathologyAnonym_Train_Valid.py* contains the training and validation processes.
* *./pathanonym_Prediction.py* all the prediction and testing processes.
* For EER calculation you should use either of the anonymization methods' folders based on your need.



------
### In case you use this repository, please cite the original paper:

Tayebi Arasteh S, Arias-Vergara T, Perez-Toro PA, et al. *The Impact of Speech Anonymization on Pathology and Its Limits*. arXiv:2404.08064 (2024).

### BibTex

    @article {pathology_anonym,
      author = {Tayebi Arasteh, Soroosh and Arias-Vergara, Tomas and Perez-Toro, Paula Andrea and Weise, Tobias and Packhäuser, Kai and Schuster, Maria and Noeth, Elmar and Maier, Andreas and Yang, Seung Hee},
      title = {The Impact of Speech Anonymization on Pathology and Its Limits},
      year = {2024},
      journal = {arXiv:2404.08064},
      doi = {10.48550/arXiv.2404.08064}
    }
