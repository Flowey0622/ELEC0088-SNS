# Structure of this project

## Chatbot_Oracle

This directory includes server and client programs, as well as the program responsible for prediction, Oracle_ML_Model.py.
'app.py' includes that connection between dialogflow and server. This file has conversation and prediction. Also there is a function for dialoflow webhook in Google Cloud. There are two file that call 'app.yaml' and 'cloudbuil.yaml', they include running method for app.py and Cloud Build respectively.

## DateSets

- London 2000-01-01 to 2024-01-31.csv
- tfl-daily-cycle-hires.csv

## Model_Training

This includes programs for training two models, GRU and LSTM. Run the two programs, Model_Training_GRU.py and Model_Training_LSTM.py, they will train the GRU and LSTM model and store the trained models in the Models_Trained directory.

## Models_Trained

This directory stores the trained models, which will be loaded on the server and applied to predict data.

## ELEC0088 SNS Assignment Final Report.pdf

The report for this project

## requirements

All the modules needed in this project

# Project Flow Chart

<img src="https://github.com/Flowey0622/Stronger/assets/160813460/d8ba7584-276f-49bb-a4d2-8c0efcbf13b8" width="550px">

# Server_with_ML

The server script establishes a multi-threaded TCP server designed to interact with clients in real-time, providing personalized weather and bike hire predictions. It utilizes Python's socket, threading, pandas, re, and numpy libraries for networking, concurrent execution, data manipulation, regular expression matching, and numerical operations, respectively.

# Client

The client script connects to the server via TCP, facilitating a two-way communication channel for users to interact with the Oracle prediction bot. It sends user inputs to the server and displays responses, creating an interactive dialogue experience.


