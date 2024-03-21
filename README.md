team members：
1. At present, the code of our machine learning model is in files Weather_Pred_LSTM.py and Weather_Pred_GRU.py. You can read Weather_Pred_Notebook.ipynb to better understand the details of the code. The code in Notebook is slightly different from that in the .py file: There is no use of a for loop, which means that only one day's data can be trained at a time (you can change the value of N to choose the day). This is easier for understanding and demonstrating, and also saves running time, There are also slightly more explanations in Notebook.
2. After considering correlation and conducting new filtering on the data as Andria suggested before, the prediction accuracy of the next day's bicycle rental data is improved a little, which is shown in Backups/BikeHire_Pred_Notebook_01.ipynb
3. In order to maintain the integrity of the overall structure, we are currently choose to train all five target features in a single model, which makes it impossible for us to optimize the model for different objectives, which I did not consider before. Next if time permits, I may consider training these target features separately and optimizing them as much as possible (which means we need to train and save 35 models, I think it might be a bit cumbersome and chaotic, but I have no other better idea). If you have any suggestions, please share them in the WhatsApp group. Thank you!

# Structure of this project
## Backups
This directory includes the methods and ideas tried during the code writing process. They are temporarily left here in case it may be useful for future improvements. This directory and its contents will be deleted when it is finally completed.

## Chatbot_Oracle
This directory includes server and client programs, as well as the program responsible for prediction, Oracle-ML_Model.py

## DateSets
- London 2000-01-01 to 2024-01-31.csv
- tfl-daily-cycle-hires.csv

## Model_Training
This includes programs for training two models, GRU and LSTM, as well as simplified versions in the form of Jupyter Notebook to facilitate understanding, testing, and improving the model.
After running the two programs, they will train the GRU and LSTM model and store the trained models in the Models_Trained directory.

## Models_Trained
This directory stores the trained models, which will be loaded on the server and applied to predict data.
