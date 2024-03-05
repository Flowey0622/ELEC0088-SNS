import pandas as pd
import numpy as np

# loading dataset into the program
dataset1 = pd.read_csv('D:\SNS\London 2000-01-01 to 2024-01-31.csv', index_col="datetime")
dataset2 = pd.read_csv(r'D:\SNS\tfl-daily-cycle-hires.csv', index_col="Day", parse_dates=["Day"])

# filter dataset1 from 2010.07.30 to 2024.01.31
start_date = pd.to_datetime('2010-07-30')
dataset1.index = pd.to_datetime(dataset1.index)
dataset1_filtered = dataset1[(dataset1.index >= start_date)]

# combining the two datasets
merged_data = pd.merge(dataset1_filtered, dataset2, left_index=True, right_index=True, how='outer')

# changing index data type into datetime
merged_data.index = pd.to_datetime(merged_data.index)
print(merged_data.index)

# Print with the merged data
pd.set_option('display.max_rows', None)  # display all rows
pd.set_option('display.max_columns', None)  # display all columns
pd.set_option('display.width', None)  # no limitation with width
print(merged_data.head())
print(merged_data.tail())
#print(merged_data)
print(merged_data.columns)

# Checking number of missing data in every column
print(merged_data.isnull().sum())

# checking percentage of missing data in a column
perc_missing_values = merged_data.isnull().sum()/merged_data.shape[0]
print(perc_missing_values)

# removing columns where null percentage is greater than 10
columns_cleaned = merged_data.columns[perc_missing_values<.1]

# changing the dataset so that it only contains required columns
merged_data=merged_data[columns_cleaned].copy()

# Filling in the missing data in columns by the previous day values
merged_data=merged_data.ffill()

# Checking again the number of missing data in every column after filling in the missing values
print(merged_data.isnull().sum())

# Checking datatypes of each column to ensure are of correct types for ML processing
print(merged_data.dtypes)

# Checking how many entries for each year to identify if there is any day weather record is missing
print(merged_data.index.year.value_counts().sort_index())

# print final dataset
final_dataset = merged_data.copy()
print(final_dataset.head())
print(final_dataset.tail())
print(final_dataset.shape)
