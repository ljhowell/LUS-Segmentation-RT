# LUS-Segmentation-RT
Deep learning for real-time multi-class segmentation of artefacts in lung ultrasound (LUS)

![LUS_examples](https://github.com/ljhowell/LUS-Segmentation-RT/assets/55801295/c52841cd-e465-4658-8c13-89379fcadfca)

## Citation
If you use this code, please cite the following paper:

TODO: Update citation with DOI etc when paper is published
```
Deep learning for real-time multi-class segmentation of artefacts in lung ultrasound, Lewis Howell, Nicola Ingram, Roger Lapham, Adam Morrell, James R. McLaughlan, Ultrasonics, 2024, XX, XXX, ISSN XXXX, https://doi.org/XXXXXXX.
```

## Introduction
Lung ultrasound (LUS) is a safe and cost-effective modality for assessing lung health. However, interpreting LUS images remains challenging due to its reliance on the interpretation of artefacts including A-lines and B-lines. We propose a U-Net deep learning model for multi-class segmentation of objects (ribs, pleural line) and artefacts (A-lines, B-lines, B-line confluence) in ultrasound images of a lung training phantom, suitable for real-time implementation.

![LUS_BLAS](https://github.com/ljhowell/LUS-Segmentation-RT/assets/55801295/17d3dcda-5f42-4aa8-b2a4-520dbdb1b6c0)

This GitHub repository includes the code for training and evaluating the model as well as a demonstration of the BLAS - a semi quantitiative metric for assessing the area of the intercostal space occupied by B-lines. The phantom data is available at XXXX. 

## Requirements:
The code was developed using Ubuntu 20.04 (but has also been tested with Windows 10), CUDA 11.4, cuDNN 8.4.1 and models implemented in TensorFlow 2.9.1 for Python 3.9. 

It is recommended to use a GPU with at least 8GB of memory for training and evaluation.

## Installation
The required python dependencies are given as a ```requirements.txt``` file or a ```Pipfile.lock``` and can be installed using pip3 or pipenv. 

If using pip3, the requirements can be installed using:
```
pip3 install -r requirements.txt
```
or if using pipenv, the requirements can be installed using:
```
pipenv install
```

# Getting started

The notebook ```LUS_Segmentation_RT.ipynb``` provides a step-by-step guide to training and evaluating the model and is the best place to start. The function ```train_model()``` in ```train_model.py``` can also be used to train the model and may be used for hyperparameter optimisation.

For real-time implementation with an ultrasound machine, the script ```segement_lus_rt.py``` can be used to segment images in real-time. This script requires the ultrasound machine to be connected to the computer via a video capture card. 

https://github.com/ljhowell/LUS-Segmentation-RT/assets/55801295/68a7ab16-b983-42f3-bb42-04a6c2bc29f8







