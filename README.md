# Bird-Classification-ML

## Overview
This repository implements a deep-learning system for classifying **200 bird species**, using transfer learning with pre-trained convolutional networks. It explores the effect of fine-tuning vs feature extraction in image classification, with emphasis on managing overfitting in a smaller dataset context.

## Folder Structure & Order
The Training Dataset and Testing Dataset are supposed to be placed inside Dataset Folder under Train Folder and Test Folder respectively.

## App.py
App.py is the main app python file that is used to create and build the AI Interface through Gradio & HuggingFace

## AI Interface URL
An AI Interface has been implemented on Hugging Face.
https://huggingface.co/spaces/Richmond28/Bird_Classification_Model

## Dataset & Preprocessing  
The dataset consists of 200 bird species and is divided into training, validation, and test sets. Key preprocessing steps include:  
- Normalization using ImageNet mean and standard deviation  
- Augmentation during training: Random Resized Crop, Horizontal Flip, Color Jitter  
- Consistent resizing and cropping for validation/test data

## Model Architectures  
Two architectures are evaluated:  
- **VGG16**: A deep convolutional neural network (~138M parameters) using stacked 3×3 convolutions and fully connected classification head. Fine-tuning allows adaptation to bird-specific features.  
- **ResNet18**: A residual network (~11M parameters) with skip connections and global average pooling, used here primarily for feature extraction by freezing the backbone and training only the classifier layer.

## Training & Evaluation  
Key details:  
- Loss Function: Cross-Entropy Loss  
- Optimizer: Adam  
- Validation monitoring with early stopping to avoid overfitting  
- Metrics used: Top-1 Accuracy, Macro Average Accuracy (equal weight per class), Confusion Matrix, Precision/Recall/F1-score

## Results  
- **VGG16**: Initial accuracy ~49.38 % → After fine-tuning: ~67 %  
- **ResNet18**: Initial accuracy ~39.79 % → After feature-extraction only: ~~51 %  
These results demonstrate that fine-tuning deeper layers significantly enhances performance in fine-grained classification tasks.
