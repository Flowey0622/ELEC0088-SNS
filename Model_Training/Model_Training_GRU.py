import pandas as pd
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

# ##============= Data loading and processing =============##
# loading dataset into the program
dataset1 = pd.read_csv('../DataSets/London 2000-01-01 to 2024-01-31.csv', index_col="datetime")
dataset2 = pd.read_csv(r'../DataSets/tfl-daily-cycle-hires.csv', index_col="Day", parse_dates=["Day"])

# define the features we want to predict
target_features = ['tempmax', 'tempmin', 'windspeed', 'precip', 'Number of Bicycle Hires']
features_unit = ['($C°$)', '($C°$)', '($m / s$)', '($mm$)', '']

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
merged_data['Number of Bicycle Hires'] = merged_data['Number of Bicycle Hires'].apply(lambda x: float(x.replace(',', '')))

# Filling in the missing data in columns by the previous day values
merged_data = merged_data.ffill()

# Analyze the read weather data and draw temperature data
plt.figure(figsize=(16, 9))
for i, feature in enumerate(target_features):
    value = merged_data.loc[:, feature]

    plt.subplot(4, 2, i + 1)
    plt.plot(range(len(value)), value)  # paint all data
    plt.xlabel('Samples')
    plt.ylabel(r'' + feature + features_unit[i])

    plt.title('Data of ' + feature + ' in the dataset')

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)
plt.show()

# ##======== Prepare the datasets required for the model ========##
# Normalise data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(merged_data)
scaled_data = pd.DataFrame(scaled_data, columns=merged_data.columns)
std = merged_data.loc[:, target_features].std(axis=0)

# Specify the ratio of training set, validation set, and test set
train_ratio = 0.7
val_ratio = 0.15
# test_ratio = 0.15

# Calculate the size of the training, validation, and testing sets
train_size = int(len(scaled_data) * train_ratio)
val_size = int(len(scaled_data) * val_ratio)

lookback = 30   # use the data of last 30 days to predict that of next day
N = 7           # forecast the weather on the Nth day

referencing_mae = []
test_mae = []
train_history = []

for forecast in range(N):
    # generate x_train and y_train
    x_train = []
    y_train = []
    for i in range(lookback + forecast, train_size + 1):
        x_train.append(scaled_data[i - lookback - forecast:i - forecast])
        y_train.append(scaled_data.loc[i, target_features])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # generate x_val and y_val
    x_val = []
    y_val = []
    for i in range(train_size + lookback + forecast, train_size + val_size + 1):
        x_val.append(scaled_data[i - lookback - forecast:i - forecast])
        y_val.append(scaled_data.loc[i, target_features])
    x_val, y_val = np.array(x_val), np.array(y_val)

    # generate x_test and y_test
    x_test = []
    y_test = []
    for i in range(train_size + val_size + lookback + forecast, len(scaled_data)):
        x_test.append(scaled_data[i - lookback - forecast:i - forecast])
        y_test.append(scaled_data.loc[i, target_features])
    x_test, y_test = np.array(x_test), np.array(y_test)

    # A common sense based, non machine learning method  is
    # used to calculate the MAE (Mean Absolute Error) of a
    # common sense based method that always predicts that
    # the next day's data is equal to the previous day's data.
    target_feature_index = [merged_data.columns.get_loc(feature) for feature in target_features]
    y_pred = np.empty(y_test.shape)
    for i in range(x_test.shape[0]):
        y_pred[i] = x_test[i, -1, target_feature_index]
    error = y_pred - y_test
    referencing_mae.append(np.mean(np.abs(error), axis=0)*std)

    # ##=========== build and train the model ===========##
    # Train and evaluate a stacked GRU model using dropout regularization
    model = Sequential()
    model.add(layers.GRU(32,
                          dropout=0.1,
                          recurrent_dropout=0.5,
                          return_sequences=True,
                          input_shape=(None, scaled_data.shape[-1])))
    model.add(layers.GRU(16,
                          activation='relu',
                          dropout=0.1,
                          recurrent_dropout=0.5))
    model.add(layers.Dense(len(target_features)))
    model.summary()

    # compile and train the model
    model.compile(optimizer=RMSprop(learning_rate=0.001),
                  loss='mae')
    history = model.fit(x_train, y_train,
                        epochs=25,
                        batch_size=128,
                        validation_data=(x_val, y_val))

    # save the model
    filename_format = 'GRU_Model_Day{}.h5'
    filename = filename_format.format(forecast+1)
    model.save('../Models_Trained/' + filename)

    # save the training history
    train_history.append(history)

    # ##================ test the model ================##
    score = model.evaluate(x_test, y_test, verbose=0)
    # denormalization the error, which shows the average error
    # measured in Celsius degree.
    error = model.predict(x_test) - y_test
    test_mae.append(np.mean(np.abs(error), axis=0)*std)

plt.figure(figsize=(16, 12))

for i, history in enumerate(train_history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))

    plt.subplot(4, 2, i+1)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title(f'Training and validation loss on Day {i+1}')
    plt.legend()

plt.tight_layout()
plt.show()

for i in range(N):
    print(f'Reference Error on Day {i+1} :')
    print(referencing_mae[i])
    print(f'Test Error on Day {i+1} :')
    print(test_mae[i])


