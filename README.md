# covid-caps-project
a capsule network based project to detect COVID19 in lung X-Rays
This project belongs to Noga Mudrik and Roman Frenkel.

This project is based on the covid-caps project:
https://github.com/ShahinSHH/COVID-CAPS

## update board
We did several changes to it:
* Pre-processed the input images to remove noises such as letters, markings, numbers, etc.
This was removed from the development plan since we chose alternative research directions.
* Added code to handle missing files we found in the pre-training dataset.
* Changed the exporting of images to numpy arrays to be in batches, 
so the memory does not overflow and the runtime is reduced.

## Steps to prepare the pre-training dataset
1)  Follow the guideline in the database folder to download and unpack the original dataset.
2)  Run xray14_preprocess.py
    This will create another folder in the current directory named database_preprocessed
    to store downscaled images all in one place.
3)  Run filterOutMissingFiles.py to align the actual data with the file lists
4)  Run xray14_selectionMT.py
    It will import preprocessed images generated in the 2nd step as numpy arrays and stack them in 500 image chunks 
    to form 2X500 numpy arrays named X_images and Y_labels. 
    These two numpy arrays will then be used to pretrain our model.
5)  run stackNPYFiles.py so we can create 2 final data and label files (the original way to do it was too long).
6)  run pre-train.py 
    (we set batch size to be 70 - this is the max for our GPU - GForce DTX 1080TI with 11 giga RAM) 
    training time for 100 epochs is about 20 hours. 

## Requirements

    Tested with tensorflow-gpu 2 and keras-gpu 2.2.4
    Python 3.6
    OpenCV
    Pandas
    Itertools
    Glob
    OS
    Scikit-image
    Scikit-learn
    Matplotlib
    Numpy

## Code

The code for the Capsule Network implementation is adapted from here. Codes to prepare the X-ray14 dataset are adopted from here. Codes are available as the following list:

    pre-train.py : Codes for pre-training
    binary-after.py : Codes for fine-tuning - baseline
    binary-after-updated.py : Codes for fine-tuning - updated loss and topology
    test_binary.py : Test and Evaluation
    weight-improvement-binary-after-44.h5 : COVID-CAPS weights after fine-tuning
    pre-train.h5 : Weights for the pre-trained model (taken from the original paper)
    pre-train_ourTrainBaseline.h5 : Weights for the pre-trained model (re-created by us - baseline)
    binary.py : Main code without pre-training
    weights-improvement-binary-86.h5 : Best model's weights without pre-training
    filterOutMissingFiles.py : aligns lists of images with the actual images that exist in the folder
    xray14_preprocess.py : Extracting and rescaling the Chest XRay14 dataset
    xray14_selectionMT.py : Converting downscaled Xray14 images into numpy arrays and saving them
    stackNPYFiles.py : creating the final 13 giga data file with corresponding labesl from the output of xray14_selectionMT.py
    

