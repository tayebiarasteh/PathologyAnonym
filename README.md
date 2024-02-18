# Automatic anonymization of pathological speech



Overview
------

* This is the official repository of the paper [**Automatic anonymization of pathological speech**](TODO).

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

Tayebi Arasteh S, Noeth E, Schuster M, et al. *Automatic anonymization of pathological speech*. ArXiv (2024)

### BibTex

    @article {pathology_anonym,
      author = {Tayebi Arasteh, Soroosh and Noeth, Elmar, and Schuster, Maria and Maier, Andreas and Yang, Seung Hee},
      title = {Automatic anonymization of pathological speech},
      year = {2024},
    }
