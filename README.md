# Structure of this project

## Backups

This directory includes the methods and ideas tried during the code writing process. They are temporarily left here in case it may be useful for future improvements. This directory and its contents will be deleted when our work is finally completed.

## Chatbot_Oracle

This directory includes server and client programs, as well as the program responsible for prediction, Oracle_ML_Model.py.

## DateSets

- London 2000-01-01 to 2024-01-31.csv
- tfl-daily-cycle-hires.csv

## Model_Training

This includes programs for training two models, GRU and LSTM. Run the two programs, Model_Training_GRU.py and Model_Training_LSTM.py, they will train the GRU and LSTM model and store the trained models in the Models_Trained directory.

You can read Model_Training_Notebook.ipynb to better understand the details of the code. The code in Notebook is slightly different from that in the .py file: There is no use of a for loop, which means that only one day's data can be trained at a time (you can change the value of N to choose the day). This is easier for understanding and demonstrating, and also saves running time, There are also slightly more explanations in Notebook.

## Models_Trained

This directory stores the trained models, which will be loaded on the server and applied to predict data.

# Project Flow Chart

<img src="https://github.com/Flowey0622/Stronger/assets/160813460/0eab988c-a9b8-42fb-ba09-d84e3eaaaffe" width="550px">
