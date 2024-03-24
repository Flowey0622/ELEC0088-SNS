import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


class MyModel:

    __MODEL_TYPE = 'GRU'
    __N = 7
    __target_features = ['tempmax', 'tempmin', 'windspeed', 'precip', 'Number of Bicycle Hires']

    def __init__(self, today):

        # load the model
        self.models = pd.DataFrame(index=range(self.__N), columns=self.__target_features)
        for feature in self.__target_features:
            for date in range(self.__N):
                filename = self.__MODEL_TYPE + f'_Model_Day{date+1}_{feature}.h5'
                self.models.at[date, feature] = load_model('../Models_Trained/' + filename)

        # load the dataset
        # define a date in test_set as "today"
        self.TODAY = today
        self.data_x = []

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

        # Normalise data
        scaled_data = pd.DataFrame(index=range(merged_data.shape[0]), columns=merged_data.columns)
        self.__scalers = pd.Series(index=merged_data.columns)
        for feature in merged_data.columns:
            scaler = MinMaxScaler()
            scaled_data[feature] = scaler.fit_transform(merged_data[feature].to_frame())
            self.__scalers[feature] = scaler

        # scaler = MinMaxScaler()
        # scaled_data = scaler.fit_transform(merged_data)
        # scaled_data = pd.DataFrame(scaled_data, columns=merged_data.columns)
        self.__deltas = merged_data.max() - merged_data.min()

        # Calculate correlation matrix
        correlation_matrix = scaled_data.corr()

        for feature in self.__target_features:

            correlation_with_target = correlation_matrix[feature].abs().sort_values(ascending=False)
            # Print the correlation results
            print("Correlation with", feature)
            print(correlation_with_target)

            # Only retain features have high correlation with the target feature
            features_used = list(correlation_with_target[correlation_with_target > 0.3].index)
            filtered_data = scaled_data.loc[:, features_used]

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

            data = np.array(filtered_data[test_start_index + delta - lookback: test_start_index + delta])
            self.data_x.append(np.expand_dims(data, axis=0))

    def predict(self, date, feature):

        i = self.__target_features.index(feature)
        pred = (self.models.at[date-1, feature].predict(self.data_x[i]))
        pred_value = self.__scalers[feature].inverse_transform(pred)
        # i = self.__target_features.index(feature)
        # value = pred.iloc[i]

        return pred_value
