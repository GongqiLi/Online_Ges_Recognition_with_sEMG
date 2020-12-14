# Online Gesture Recognition with sEMG

## Introduction
This project is for the real-time gesture classification, 
and potentially in the future, real-time gesture visualization. It applies two classifiers,
i.e. XGB and Deep Neural Network for the classification task and is compatible for both
pre-recorded data and real-time data as input. A detailed explanation of the functionality
of each module is given in the 'Project Structure' section.  

## Installation
Steps for the environment setup:
##### 
    conda config --add channels conda-forge
    conda install numpy==1.19.2
    conda install pytorch -c pytorch
    conda install pandas
    conda install scipy
    conda install matplotlib
    conda install openpyxl
    conda install pyqtgraph
    conda install -c anaconda pyqt
Note that xgb-related packages require extra installation steps. 

## Project Structure
#### real_time_ui.py
Real Time User Interface: Contains the PyQt Main Frame for visualizing the result 
from the classification and the module that import real-time data from the armband. 
Also, the user can change the parameter 'real-time' to False and specify the data
location if want to import pre-recorded data instead. 

#### online_ges_detection_dnn.py 
The finite state machine that stores the time series and output the classification 
results using deep neural network given sEMG inputs. 

#### online_ges_detection_xgb.py 
The finite state machine that stores the time series and output the classification 
results using extreme gradient boost given sEMG inputs. 

#### dnn_model.py
The class that specifies parameters of the deep neural network. Change this one for hyper-parameter
search. 
    
#### dnn_train.py
This module undertake the task to train the deep neural network with 'dnn.pt' as output. 
Run this before 'real_time_ui.py' if the dnn has not been trained. 

#### xgb_train.py
This module undertake the task to train the xgb model with 'xgb.model' as output. 
Run this before 'real_time_ui.py' if the xgb model has not been trained. 

#### FetchGForceData64.dll & gforce64d.dll
Files needed to take in real-time data from the armband.



