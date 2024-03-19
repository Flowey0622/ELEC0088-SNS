import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

class MyModel:

    __MODEL_TYPE = 'GRU'
    __N = 7
    __target_features = ['tempmax', 'tempmin', 'windspeed', 'precip', 'Number of Bicycle Hires']

    def __init__(self, today):
        # define a date in test_set as "today"
        self.TODAY = today
        # ##============= Data loading and processing =============##
        # loading dataset into the program
        dataset1 = pd.read_csv('../DataSets/London 2000-01-01 to 2024-01-31.csv', index_col="datetime")
        dataset2 = pd.read_csv(r'../DataSets/tfl-daily-cycle-hires.csv', index_col="Day", parse_dates=["Day"])

        # filter dataset1 from 2010.07.30 to 2024.01.31 to fit with dataset2
        start_date = pd.to_datetime('2010-07-30')
        dataset1.index = pd.to_datetime(dataset1.index)
        dataset1_filtered = dataset1[(dataset1.index >= start_date)]

        # delete useless columns and columns with too many NaN values
        dataset1_filtered.drop(columns=['name', 'severerisk', 'windgust',
                                        'preciptype', 'precipprob', 'solarradiation',
                                        'solarenergy', 'uvindex', 'sunrise',
                                        'sunset', 'conditions', 'description',
                                        'icon', 'stations', 'sunset'],
                               inplace=True)

        # combining the two datasets
        merged_data = pd.merge(dataset1_filtered, dataset2, left_index=True, right_index=True, how='outer')

        # change the values of bike hires to float
        merged_data['Number of Bicycle Hires'] = merged_data['Number of Bicycle Hires'].apply(
            lambda x: float(x.replace(',', '')))

        # Filling in the missing data in columns by the previous day values
        merged_data = merged_data.ffill()

        # ##======== Prepare the dataset(x_test) required for the model ========##
        # Normalise data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(merged_data)
        scaled_data = pd.DataFrame(scaled_data, columns=merged_data.columns)
        self.__std = merged_data.loc[:, self.__target_features].std(axis=0)

        # Specify the ratio of training set, validation set, and test set
        train_ratio = 0.7
        val_ratio = 0.15

        # Calculate the size of the training, validation, and testing sets
        train_size = int(len(scaled_data) * train_ratio)
        val_size = int(len(scaled_data) * val_ratio)

        lookback = 30  # use the data of last 30 days to predict that of next day
        N = 7  # forecast the weather on the Nth day
        forecast = N - 1

        # find the index of the set 'today'
        test_start_index = train_size + val_size + lookback
        test_start_date = pd.to_datetime('2022-2-19')
        delta = (self.TODAY - test_start_date).days

        self.x_test = np.array(scaled_data[test_start_index + delta - lookback: test_start_index + delta])
        self.x_test = np.expand_dims(self.x_test, axis=0)

        # load the model
        self.models = []
        for i in range(1, N+1):
            filename_format = self.__MODEL_TYPE + '_Model_Day{}.h5'
            filename = filename_format.format(i)
            self.models.append(load_model('../Models_Trained/' + filename))

    def predict(self, date, feature):

        pred = (self.models[date - 1].predict([self.x_test])).reshape(len(self.__target_features))
        pred = pred * self.__std
        i = self.__target_features.index(feature)
        value = pred.iloc[i]

        return value
