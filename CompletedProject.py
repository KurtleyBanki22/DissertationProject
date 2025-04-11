# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
import os 
import sys
import matplotlib.pyplot as plt
import datetime as dt
import math
import seaborn as sns

# Load the data
df = pd.read_csv("C:/Users/User/Downloads/archive (8)/household_power_consumption.csv", sep=";")

#print(df.head())

na_sum = df.isna().sum()
#print(na_sum)

#print(df.value_counts)
#print(df.describe)

null_rows = df[df.isnull().any(axis=1)]
#print(null_rows)

df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')
df['Global_reactive_power'] = pd.to_numeric(df['Global_reactive_power'], errors='coerce')
df['Voltage'] = pd.to_numeric(df['Voltage'], errors='coerce')
df['Global_intensity'] = pd.to_numeric(df['Global_intensity'], errors='coerce')
df['Sub_metering_1'] = pd.to_numeric(df['Sub_metering_1'], errors='coerce')
df['Sub_metering_2'] = pd.to_numeric(df['Sub_metering_2'], errors='coerce')

#print(df.dtypes)

null_rows = df[df.isnull().any(axis=1)]
#print(null_rows)

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)
df['Year'] = df['DateTime'].dt.year
df['Month'] = df['DateTime'].dt.month
df['Day'] = df['DateTime'].dt.day
df['Hour'] = df['DateTime'].dt.hour
df['Minute'] = df['DateTime'].dt.minute
df

def is_holiday(date):
    year = int(date.split('/')[2])
    month = int(date.split('/')[1])
    day = int(date.split('/')[0])
    
    if dt.datetime(year, month, day).weekday() >= 5:
        return 1.0
    else:
        return 0.0

for data in [df]:
    data['Is_holiday'] = data['Date'].apply(lambda x: is_holiday(x))
    data['Light']      = data['Time'].apply(lambda x: 1.0 if int(x[:2]) >= 6 and int(x[:2]) < 18 else 0.0)
    data['Time']       = data['Time'].apply(lambda x: (int(x[:2]) * 60.0 + int(x[3:5])) / 1440.0)
    
    #print(df.columns)
    
df.drop(['Date'], axis=1, inplace=True)
df.drop(['DateTime'], axis=1, inplace=True)

#print(df.describe)

df = df[df['Minute'] % 10 == 0].copy()


df.dropna(inplace=True)
print(null_rows)

df.to_csv('dataset2.csv', index=False)

file_path ="C:/Users/User/Documents/DataV/dataset2.csv"

df = pd.read_csv(file_path)

print(df.head())

plt.figure(figsize=(12, 6))
plt.plot(df['Time'], df['Global_active_power'])
plt.xlabel('Time')
plt.ylabel('Global Active Power')
plt.title('Global Active Power over Time')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['Global_intensity'], bins=20)
plt.xlabel('Global Intensity')
plt.ylabel('Frequency')
plt.title('Distribution of Global Intensity')
plt.show()

sub_metering_cols = ['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
plt.figure(figsize=(8, 6))
df[sub_metering_cols].boxplot()
plt.xlabel('Sub-metering')
plt.ylabel('Value')
plt.title('Box Plot of Sub-metering Measurements')
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Global_active_power'], df['Voltage'])
plt.xlabel('Global Active Power')
plt.ylabel('Voltage')
plt.title('Global Active Power vs. Voltage')
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(df['Voltage'], bins=20)
plt.xlabel('Voltage')
plt.ylabel('Frequency')
plt.title('Distribution of Voltage')
plt.show()

plt.figure(figsize=(8, 6))
df.boxplot(column='Global_active_power', by='Month')
plt.xlabel('Month')
plt.ylabel('Global Active Power')
plt.show()

variables = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']
sns.pairplot(df[variables])
plt.show()

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.subplots_adjust(wspace=0.2)
sns.boxplot(x='Year', y='Global_active_power', data=df)
plt.xlabel('Year')
plt.title('Box plot of Yearly Global Active Power')
sns.despine(left=True)
plt.tight_layout()

from scipy import stats
from statsmodels.tsa.stattools import adfuller
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
df['Global_active_power'].hist(bins=50)
plt.title('Global Active Power Distribution')

plt.subplot(1,2,2)
stats.probplot(df['Global_active_power'],plot=plt)

plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
df.groupby('Year').Global_active_power.agg('mean').plot()
plt.xlabel('')
plt.title('Mean global active power by year')



plt.subplot(2,2,3)
df.groupby('Month').Global_active_power.agg('mean').plot()
plt.xlabel('')
plt.title('Mean global active power by month')

plt.subplot(2,2,4)
df.groupby('Day').Global_active_power.agg('mean').plot()
plt.xlabel('')
plt.title('Mean global active power by day')

pd.pivot_table(df.loc[df['Year']!=2006],values='Global_active_power',columns='Year',index='Month').plot(subplots=True,figsize=(12,12),layout=(3,5),sharey=True)



from sklearn.model_selection import train_test_split

X = df.drop('Global_active_power', axis=1)
y = df['Global_active_power']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train


from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)
linear_reg_predictions = linear_reg.predict(X_test)
linear_reg_rmse = np.sqrt(mean_squared_error(y_test, linear_reg_predictions))
linear_reg_rmse

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')

plt.scatter(X_test['Global_intensity'], linear_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('Linear Regression - Actual Values and Predictions')
plt.legend()
plt.show()


from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor()
decision_tree_reg.fit(X_train, y_train)
decision_tree_reg_predictions = decision_tree_reg.predict(X_test)
decision_tree_reg_rmse = np.sqrt(mean_squared_error(y_test, decision_tree_reg_predictions))
decision_tree_reg_rmse

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')

plt.scatter(X_test['Global_intensity'], decision_tree_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('DecisionTreeRegressor - Actual Values and Predictions')
plt.legend()
plt.show()


from sklearn.ensemble import RandomForestRegressor
random_forest_reg = RandomForestRegressor()
random_forest_reg.fit(X_train, y_train)
random_forest_reg_predictions = random_forest_reg.predict(X_test)
random_forest_reg_rmse = np.sqrt(mean_squared_error(y_test, random_forest_reg_predictions))
random_forest_reg_rmse

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')
plt.scatter(X_test['Global_intensity'], random_forest_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('RandomForestRegressor - Actual Values and Predictions')
plt.legend()
plt.show()



from sklearn.neural_network import MLPRegressor
mlp_reg = MLPRegressor()
mlp_reg.fit(X_train, y_train)
mlp_reg_predictions = mlp_reg.predict(X_test)
mlp_reg_rmse = np.sqrt(mean_squared_error(y_test, mlp_reg_predictions))
mlp_reg_rmse

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')
plt.scatter(X_test['Global_intensity'], mlp_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('MLPRegressor - Actual Values and Predictions')
plt.legend()
plt.show()


from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

lgbm_reg = LGBMRegressor()
lgbm_reg.fit(X_train, y_train)
lgbm_reg_predictions = lgbm_reg.predict(X_test)
lgbm_reg_rmse = np.sqrt(mean_squared_error(y_test, lgbm_reg_predictions))
lgbm_reg_rmse

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')

plt.scatter(X_test['Global_intensity'], lgbm_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('LGBMRegressor - Actual Values and Predictions')
plt.legend()
plt.show()


xgb_reg = XGBRegressor()
xgb_reg.fit(X_train, y_train)
xgb_reg_predictions = xgb_reg.predict(X_test)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_test, xgb_reg_predictions))
xgb_reg_rmse

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')

plt.scatter(X_test['Global_intensity'], xgb_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('XGBRegressor - Actual Values and Predictions')
plt.legend()
plt.show()


grad_boost_reg = GradientBoostingRegressor()
grad_boost_reg.fit(X_train, y_train)

ada_boost_reg = AdaBoostRegressor()
ada_boost_reg.fit(X_train, y_train)

bagging_reg = BaggingRegressor()
bagging_reg.fit(X_train, y_train)

grad_boost_reg_predictions = grad_boost_reg.predict(X_test)
ada_boost_reg_predictions = ada_boost_reg.predict(X_test)
bagging_reg_predictions = bagging_reg.predict(X_test)

grad_boost_reg_rmse = np.sqrt(mean_squared_error(y_test, grad_boost_reg_predictions))
ada_boost_reg_predictions_rmse = np.sqrt(mean_squared_error(y_test, ada_boost_reg_predictions))
bagging_reg_predictions_rmse = np.sqrt(mean_squared_error(y_test, bagging_reg_predictions))

print("grad_boost_reg_rmse  {}".format(grad_boost_reg_rmse))
print("ada_boost_reg_predictions_rmse  {}".format(ada_boost_reg_predictions_rmse))
print("bagging_reg_predictions_rmse  {}".format(bagging_reg_predictions_rmse))

rmse_values = [linear_reg_rmse, 
               decision_tree_reg_rmse, 
               random_forest_reg_rmse,
               mlp_reg_rmse,
              lgbm_reg_rmse,
              xgb_reg_rmse,
              grad_boost_reg_rmse,
              ada_boost_reg_predictions_rmse,
              bagging_reg_predictions_rmse]

models = [     'linear', 
               'decision_tree', 
               'random_forest', 
               'mlp',
              'lgbm',
              'xgb',
              'grad_boost',
              'ada_boost_reg_predict',
              'bagging_reg_predict'
         ]

plt.bar(models, rmse_values)
plt.xlabel('Model')
plt.ylabel('RMSE')
plt.title('RMSE Values')
plt.xticks(rotation=90)


for i in range(len(models)):
    plt.text(i, rmse_values[i], str(round(rmse_values[i], 2)), ha='center', va='bottom')

plt.show()


from xgboost import XGBRegressor

xgb_reg = XGBRegressor(learning_rate=0.1, max_depth=5, n_estimators=1000)

xgb_reg.fit(X_train, y_train)

xgb_reg_predictions = xgb_reg.predict(X_test)
xgb_reg_rmse = np.sqrt(mean_squared_error(y_test, xgb_reg_predictions))

print("XGBRegressor RMSE Values:", xgb_reg_rmse)

plt.scatter(X_test['Global_intensity'], y_test, color='blue', label='Actual Values')
plt.scatter(X_test['Global_intensity'], xgb_reg_predictions, color='red', label='Predictions')

plt.xlabel('Global_intensity')
plt.ylabel('Global_active_power')
plt.title('XGBRegressor - Actual Values and Predictions')
plt.legend()
plt.show()



data = df.copy()
data['ds'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']])
data = data.rename(columns={'Global_active_power': 'y'})

from prophet import Prophet

train_size = int(len(data) * 0.8)  # 80% of data for training
train_df = data[:train_size]
test_df = data[train_size:]


model = Prophet()
model.fit(train_df)


future_dates_365days = model.make_future_dataframe(periods=365)
future_dates_1825days = model.make_future_dataframe(periods=1825)

predictions_365days = model.predict(future_dates_365days)
predictions_1865days = model.predict(future_dates_1825days)

fig, ax = plt.subplots(figsize=(10, 6))
model.plot(predictions_365days, ax=ax)
plt.title('365-Day Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
model.plot(predictions_1865days, ax=ax)
plt.title('5-Year Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

data_copy = data.copy()

train_size = int(len(data_copy) * 0.8)  # 80% of data for training
train_df = data_copy[:train_size]
forecast_df = data_copy[train_size:]

model = Prophet()
model.fit(train_df)

future_dates_365days = model.make_future_dataframe(periods=365, freq='D')
future_dates_1825days = model.make_future_dataframe(periods=1825, freq='D')

predictions_365days = model.predict(future_dates_365days)
predictions_1825days = model.predict(future_dates_1825days)

fig, ax = plt.subplots(figsize=(10, 6))
model.plot(predictions_365days, ax=ax)
plt.title('365-Day Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
model.plot(predictions_1825days, ax=ax)
plt.title('5-Year Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

data.describe
