import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# loading dataset into the program
dataset1 = pd.read_csv('/Users/edithi/Desktop/London 2000-01-01 to 2024-02-17.csv', index_col="datetime")
print(dataset1.head())

# Checking for missing data
print(dataset1.isnull().sum())

# checking percentage of missing values in a column
perc_missing_values = dataset1.isnull().sum()/dataset1.shape[0]
print(perc_missing_values)

# removing columns where null percentage is greater than 10
columns_cleaned = dataset1.columns[perc_missing_values<.1]

# changing the dataset so that it only contains required columns
dataset1=dataset1[columns_cleaned].copy()

# Checking if there are missing data in columns and filling in
# dataset1=dataset1.ffill

# Checking datatypes of each columns and making sure are of correct data types for ML processing
print(dataset1.dtypes)
print(dataset1.index)

# Converting date from object to datetime datatype
dataset1.index = pd.to_datetime(dataset1.index)
print(dataset1.index)

# Checking for missing values within the time period
print(dataset1.index.year.value_counts().sort_index())

# dataset1["temp_max"].plot()
# plt.show()

print(dataset1.describe())
print(dataset1.shape)

dataset1.drop(columns=["name"],inplace=True)
print(dataset1.shape)
print(dataset1.head())

# dataset2 = pd.read_csv('/Users/edithi/Desktop/seattle-weather.csv')
# final_dataset = pd.concat([dataset1,dataset2])
# print(final_dataset.shape)


