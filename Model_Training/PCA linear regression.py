import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt

from sklearn.model_selection import learning_curve


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
merged_data.drop('name', axis=1, inplace=True)
final_dataset = merged_data.copy()
print(final_dataset.head())
print(final_dataset.tail())
#print(final_dataset.shape)

# Assuming ' Number of Bicycle Hires' is target variable
target_variable = 'Number of Bicycle Hires'
final_dataset['Number of Bicycle Hires'] = final_dataset['Number of Bicycle Hires'].str.replace(',', '').astype(float)

# Calculate correlation matrix
correlation_matrix = final_dataset.corr()

# Extract correlations with the target variable
correlation_with_target = correlation_matrix[target_variable].abs().sort_values(ascending=False)

# Print the correlation results
print("Correlation with", target_variable)
print(correlation_with_target)

final_dataset.drop(columns=['sunrise', 'sunset', 'conditions', 'description', 'icon', 'stations'], inplace=True)

# define features
excluded_columns = list(correlation_with_target[correlation_with_target < 0.6].index)
features = [col for col in final_dataset.columns if col not in excluded_columns]
#print(features)

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
#print(principal_data)

# use date index not number index
date_index = final_dataset.index
principal_data.index = date_index[:len(principal_data)]
final_data = pd.concat([principal_data, final_dataset[['Number of Bicycle Hires']]], axis=1)
#print(final_data)
print(pca.explained_variance_ratio_)

# figure out
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

#plt.show()

#PCA in ML
# split dataset to train list and test list
principal_data_train, principal_data_test, y_train, y_test = train_test_split(principal_data, y, test_size=0.2, random_state=42)

# use linear regression to train the model
model = LinearRegression()

# fitting model
model.fit(principal_data_train, y_train)

# Predicting the Test set results
y_predict = model.predict(principal_data_test)

# Evaluating the Model
mse = mean_squared_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)

print(f"MSE: {mse}")
print(f"R^2: {r2}")

# Calculate residuals (differences between actual and predicted values)
loss = y_test - y_predict

loss_dataset = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten(), 'Loss': loss.flatten()})

print(loss_dataset.head())
print(loss_dataset.tail())

# Extract the serial number and loss value
index_values = loss_dataset.index
loss_values = loss_dataset['Loss']

plt.figure(figsize=(10, 6))
plt.plot(index_values, loss_values, marker='.', markersize=5, linestyle='-')  # 调整markersize参数
plt.xlabel('Index')
plt.ylabel('Loss')
plt.title('Loss Variation')
plt.grid(True)
plt.show()


# cross validation
# if Cross-validated R² is close to Average R²---better generalization ability
scores = cross_val_score(model, principal_data, y, cv=5, scoring='r2')

print(f"Cross-validated R² scores: {scores}")
print(f"Average R² score: {scores.mean()}")


# draw learning curve
#def plot_learning_curve(estimator, x, y, title="Learning Curve", ylim=None, cv=None, n_jobs=None,
                        #train_sizes=np.linspace(.1, 1.0, 5)):
    #plt.figure()
    #plt.title(title)
    #if ylim is not None:
        #plt.ylim(*ylim)
    #plt.xlabel("Training examples")
    #plt.ylabel("Score")

    #train_sizes, train_scores, test_scores = learning_curve(estimator, x, y, cv=cv, n_jobs=n_jobs,
                                                            #train_sizes=train_sizes)

    #train_scores_mean = np.mean(train_scores, axis=1)
    #test_scores_mean = np.mean(test_scores, axis=1)

    #plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    #plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    #plt.legend(loc="best")
    #return plt


# 绘制学习曲线
#plot_learning_curve(model, principal_data_train, y_train, ylim=(0.0, 1.01), cv=5, n_jobs=4)
#plot_learning_curve(model, principal_data_train, y_train.ravel(), ylim=(0.0, 1.01), cv=5, n_jobs=1)

#plt.show()

