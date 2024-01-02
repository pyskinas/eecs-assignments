# Assignments for Deep Learning for Computer Vision (EECS 498/598) Winter 2022
Hi, this repo contains my completed assignments for the Univerity of Michigan's computer vision course led by Justin Johnson.
[Materials](https://web.eecs.umich.edu/~justincj/teaching/eecs498/WI2022/schedule.html) include lectures, readings and assignments.
Should you wish to check the assignments, you can clone this repo, upload it to google drive (or change the directories in the first few `.ipynb` cells) 
and run it.

I also made accompanying [lecture notes](https://pyskinas.github.io/eecs/).

# About the Assignments
These assignments all have the following details:
- They have multiple parts.
- Each part has a prewritten `.ipynb` notebook, which makes sure the code you write in the associated `.py` file runs as it should.
- The code you write in the `.py` file is usually the implementation of functions called in the `.ipynb` and `.py` files. Function, variable and return value names are given, but all technical details are implemented by the student.
- `torch.nn` was not used for Assignments 1 to 3, to develop mathematical understanding. It was used from Assignment 4.

Other details: 
- The description of each assignment is taken from the course website. 
- The assignments are from Winter 2022, however the recorded lectures were from Winter 2019 and so the online material does not cover _all_ of the knowledge required for the assignments,
  some independent research is required.

## Assignment 1 - Pytorch 101 and a kNN Classifier
In this assignment, you will first learn how to use PyTorch on Google Colab environment. You will then practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor, and finally will learn how to use Autograder for evaluating what you implement. The goals of this assignment are as follows:
- Develop proficiency with PyTorch tensors
- Gain experience using notebooks on Google Colab
- Understand the basic Image Classification pipeline and the data-driven approach (train/predict stages)
- Understand the train/val/test splits and the use of validation data for hyperparameter tuning
- Implement and apply a k-Nearest Neighbor (kNN) classifier
- Learn how to test your implementation on Autograder

## Assignment 2 - Linear Classifiers and Two-Layer Neural Network
In this assignment, you will implement various image classification models, based on the SVM / Softmax / Two-layer Neural Network. The goals of this assignment are as follows:
- Implement and apply a Multiclass Support Vector Machine (SVM) classifier
- Implement and apply a Softmax classifier
- Implement and apply a Two-layer Neural Network classifier
- Understand the differences and tradeoffs between these classifiers
- Understand how a Two-layer Neural Network can approximate an arbitrary function
- Practice implementing vectorized gradient code by checking against naive implementations, and using numeric gradient checking

## Assignment 3 - Fully Connected Neural Network and Convolutional Neural Network
In this assignment, you will implement Fully-Connected Neural Networks and Convolutional Neural Networks for image classification models. The goals of this assignment are as follows:
- Understand Neural Networks and how they are arranged in layered architectures
- Understand and be able to implement modular backpropagation
- Implement various update rules used to optimize Neural Networks
- Implement Batch Normalization for training deep networks
- Implement Dropout to regularize networks
- Understand the architecture of Convolutional Neural Networks and get practice with training these models on data

## Assignment 4 - One-Stage Detector and Two-Stage Detector
In this assignment, you will implement two different object detection systems. The goals of this assignment are:
- Learn about a typical object detection pipeline: understand the training data format, modeling, and evaluation.
- Understand how to build two prominent detector designs: one-stage anchor-free detectors, and two-stage anchor-based detectors.

Some of the code provided in the couse website does not work for this assignment.

## Assignments 5 - Image Captioning with RNNs and Transformer Model for Simple Arithmetic Operations
_Transformer part of assignmnet is incomplete due to external errors._ 

In this assignment, you will implement two different attention-based models, RNN and Transformers. The goals of this assignment are:
- Understand and implement recurrent neural networks
- See how recurrent neural networks can be used for image captioning
- Understand how to augment recurrent neural networks with attention
- Understand and implement different building blocks of the Transformer model
- Use the Transformer model on a toy dataset


## Assignment 6 - To Be Completed By 31st January
