# depth-to-ODBA-ANN
Predicting white shark's ODBA based on dive depth using artificial neural network (ANN) model

## Introduction
Overall dynamic body acceleration (ODBA) serves as a common proxy of locomotory energy expenditure for white sharks, which is the major index to understand the correlcation between white shark's behavior and energy costs. This repository contains the data of two white sharks' time-series dive depth in corresponding with ODBA metrics and the artificial neural network (ANN) model we developed in the publication "Liu, Z.Y.-C., Moxley, J.H. et al. (2019) Deep learning accurately predicts white shark locomotor activity from depth data". ANN models can be trained to predict ODBA from univariate depth (pressure) data from two free-swimming white sharks. The results in our paper can be reproduced from the code in this repository. You can also use the code for your own data. This technique can potentially be applied to other species as well. 

## Data
Two white shark dive depth data are provided: `shark_1_data.csv` and `shark_2_data.csv`. Data columns are: time (sec), depth (m), ODBA. Each dataset has 25,200 data points; 1 data point is 1 second measurement. First hour of data are used for training. The following 6 hours of data are for validation (outside/dev data). 

## Code
The script is in Python. ANN models are built with Keraas and TensorFlow backend. Main srcript is `main.py` where you can adjust model paramsters, such as moving window size, number of training epoch and batch size. ANN model building is in `ANN_train.py` where you can adjust the architecture of neural nets, including number of hidden layers and neurals as well as activation function and optimizer. `smooth.py` and `window_size.py` are the utility scripts, which perform smooth function and apply moving window size to create training data.

After running the main script, the R2 metrics will be outputted to a csv file, and the plots of training and prediction will be generated.

## Requirements
- [Anaconda / Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow 1.12](https://www.tensorflow.org/)
- [Keras 2.2](https://keras.io/)
