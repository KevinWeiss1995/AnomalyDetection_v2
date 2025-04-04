# Anomaly Detection in Network Traffic

## Overview

This project trains a binary classifier to determine abnormalities in network traffic. A variety of techniques are used to improve performance such as K-Fold cross validation, 
natural gradient descent (via KFAC), and a cyclic learning rate. The model acheives ~90% accuracy leaving a small amount of room for improvement. The model is trained on the NF-CSE-CIC-IDS2018-v3 dataset. The description is as follows:

The original pcap files of the CSE-CIC-IDS2018 dataset are utilised to generate a NetFlow-based dataset called NF-CSE-CIC-IDS2018. The total number of flows is 20,115,529 out of which 2,600,903 (12.93%) are attack samples and 17,514,626 (87.07%) are benign ones, the table below represents the dataset's distribution.

Class	Count	Description
- Benign	17,514,626	Normal unmalicious flows
- BruteForce	575,194	A technique that aims to obtain usernames and password credentials by accessing a list of predefined possibilities
- Bot	207,703	An attack that enables an attacker to remotely control several hijacked computers to perform malicious activities.
- DoS	302,966	An attempt to overload a computer system's resources with the aim of preventing access to or availability of its data.
- DDoS	1,324,350	An attempt similar to DoS but has multiple different distributed sources.
- Infiltration	188,152	An inside attack that sends a malicious file via an email to exploit an application and is followed by a backdoor that scans the network for other vulnerabilities
- Web Attacks	2,538	A group that includes SQL injections, command injections and unrestricted file uploads

Source: https://staff.itee.uq.edu.au/marius/NIDS_datasets/

## Preprocessing

The preprocessing script can be found in the preprocessing folder, and handles the following tasks:
- Removes identifying and metadata features (Flow ID, IPs, ports, etc.)
- Drops columns containing all zeros
- Converts labels to binary classification (Normal/Attack)
- Handles class imbalance using RandomUnderSampler
- Removes low-importance features using RandomForest feature importance
- Scales data using QuantileTransformer
- Splits data into train/test sets
- Saves processed datasets and transformer for later use

## Model

The model is a neural network that consists of 13 layers: 

- 5 Dense layers
- 4 Dropout layers
- 2 BatchNorm layers
- 1 Add layer (for residual connection)
- 1 Input layer

The architecture includes a residual block (layers 6-10). Residual connections were originally used to help with gradient flow during training, but our KFAC optimizer handles this well. We still implement them because they create ensemble-like behavior where the network can choose to use or bypass certain layers and provide additional paths for information flow, regardless of optimizer.

## Training

### KFAC (Kronecker-Factored Approximate Curvature)
KFAC is an optimization technique that approximates natural gradient descent. While traditional optimizers like Adam or SGD only use first-order information (gradients), KFAC uses second-order information (curvature) to make better update steps:

- Traditional optimizers might zigzag in steep valleys of the loss landscape
- KFAC is a form of natural gradient descent, which takes advantage of the probabilistic nature of neural network parameters
- The parameters are treated as a probability distribution over the parameter space
- KFAC approximates the Fisher Information Matrix using Kronecker products
- This gives better estimates of the optimal step direction
- Result: Faster convergence and better training stability

The tradeoff is increased computational cost per step, but this is often offset by needing fewer steps overall.

### Other Training Techniques
1. 5-Fold Cross Validation: The dataset is split into 5 parts, where we train on 4 parts and validate on the remaining part, rotating through all combinations. This gives us a robust estimate of model performance and helps detect overfitting.

2. Cyclic Learning Rate (CLR): Instead of using a fixed learning rate, we cycle between 0.0001 and 0.001, which helps escape local minima and often leads to better convergence. The learning rate oscillates over a fixed step size of 2000 batches.

3. Early Stopping: Training automatically stops when validation loss stops improving, preventing overfitting. The best weights are restored.

The final model is trained on the full dataset after validation confirms the architecture's effectiveness.


## Results

The model achieves strong performance:

- Overall Accuracy: 90%
- Normal Traffic (Class 0):
  - Precision: 88%
  - Recall: 94%
  - F1-Score: 91%
- Attack Traffic (Class 1):
  - Precision: 92%
  - Recall: 85%
  - F1-Score: 89%

The confusion matrix showed:
- True Negatives (Normal correctly identified): 155,706
- False Positives (Normal misclassified as Attack): 9,939
- False Negatives (Attack misclassified as Normal): 20,875
- True Positives (Attack correctly identified): 119,889

These results indicate the model is well-balanced between classes, with slightly better performance at identifying normal traffic (94% recall) than attacks (85% recall). The high precision for attack detection (92%) means we have relatively few false alarms.

## Testing

The repo includes a script for testing the model on a sample of data. You can generate a sample of data by running traffic_generator.py and traffic_modifier.py simultaneously. Below is an example of the output of use_model.py:
