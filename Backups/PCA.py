import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

# loading dataset into the program
dataset1 = pd.read_csv('../DataSets/London 2000-01-01 to 2024-01-31.csv', index_col="datetime")
dataset2 = pd.read_csv(r'../DataSets/tfl-daily-cycle-hires.csv', index_col="Day", parse_dates=["Day"])

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
columns_cleaned = merged_data.columns[perc_missing_values < 0.1]

# changing the dataset so that it only contains required columns
merged_data = merged_data[columns_cleaned].copy()

# Filling in the missing data in columns by the previous day values
merged_data = merged_data.ffill()

# Checking again the number of missing data in every column after filling in the missing values
print(merged_data.isnull().sum())

# Checking datatypes of each column to ensure are of correct types for ML processing
print(merged_data.dtypes)

# Checking how many entries for each year to identify if there is any day weather record is missing
print(merged_data.index.year.value_counts().sort_index())

# print final dataset
merged_data.drop('name', axis=1, inplace=True)
final_dataset = merged_data.copy()
print('Print the final dataset:')
print(final_dataset.head())
print(final_dataset.tail())
print(final_dataset.shape)

# define features
excluded_columns = ["sunrise", "sunset", "conditions", "description", "icon", "stations", "datetime", "Number of Bicycle Hires"]
features = [col for col in final_dataset.columns if col not in excluded_columns]
# Separating out the features
x = final_dataset.loc[:, features].values

# Separating out the target
y = final_dataset.loc[:,['Number of Bicycle Hires']].values

# Standardizing the features
X = StandardScaler().fit_transform(x)
#print(x)
#print(y)
#print(X)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X)
#print(principalComponents)

principal_data = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
#print(principalDf)

# use date index not number index
date_index = final_dataset.index
principal_data.index = date_index[:len(principal_data)]
final_data = pd.concat([principal_data, final_dataset[['Number of Bicycle Hires']]], axis=1)
#print(final_data)

# figure out
final_data['Number of Bicycle Hires'] = final_data['Number of Bicycle Hires'].str.replace(',', '').astype(float)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=10)
ax.set_ylabel('Principal Component 2', fontsize=10)
ax.set_title('2 component PCA with Number of Bicycle Hires', fontsize=10)

# use color bar 'Number of Bicycle Hires' to sign point
point = ax.scatter(
    final_data['principal component 1'],
    final_data['principal component 2'],
    c=final_data['Number of Bicycle Hires'],
    cmap='viridis',
    s=5,
)

# define color bar
color_bar = plt.colorbar(point)
color_bar.set_label('Number of Bicycle Hires')

plt.show()



