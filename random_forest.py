#!/usr/bin/env python

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pytz
from datetime import datetime

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression



# Read the regions -----------
CircadianRegions = 'CircadianRegions_2kb.csv'
CircadianRegions = pd.read_csv(CircadianRegions)

#********************************************************
#
#           TESTING ONLY!!! (comment!)
#
#CircadianRegions = CircadianRegions.sample(frac=0.1, replace=False, random_state=5461) # Sample a fraction of the df
#
#********************************************************

# Prepare data -----------
data_df = CircadianRegions[['gene','JTK_adjphase','region2kb']] # keep relevant cols
data_df = data_df.sort_values('gene') # make sure df is sorted by gene
data_df = data_df.dropna(subset=['JTK_adjphase'])

# Get features -----------

# Use an 8bp window and 3bp overlap
window_size = 8
overlap = 3
non_overlap = window_size-overlap
regions = data_df['region2kb']
# Slide window and get features
Features = regions.apply(lambda x: [x[i:i+window_size] for i in range(0, len(x)-7, non_overlap)])
Features.name = "features" # Rename
# Get unique features
unique_features = [item for sublist in Features for item in sublist]
unique_features = set(unique_features)

print("before encoding")


# One hot encoding -----------
# Create features df
features_df = pd.concat([data_df['gene'], Features], axis=1)
# Set and fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
one_hot = mlb.fit_transform(features_df["features"]) # fit
# Create a new df with the one-hot encoding and the gene column
one_hot_df = pd.DataFrame(one_hot, columns=mlb.classes_)
one_hot_df["gene"] = features_df["gene"].values
one_hot_df = one_hot_df[["gene"] + list(mlb.classes_)] # "gene" col to front
one_hot_df = one_hot_df.set_index('gene') # Set gene as index
feature_names = np.array(one_hot_df.columns) # get feature names

print("hot encoding done")

# Split data for training -----------

# Data for models
X_features = one_hot_df.iloc[:, 2:].values
Y_target = data_df["JTK_adjphase"].values
gene_names = data_df["gene"].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_features, Y_target, range(len(Y_target)), test_size=0.2, random_state=5461)

# Random Forest Model
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=5461, verbose=3)
print("before training")
rf.fit(X_train, y_train)
print("training done")
# Predicting the values for the test set
y_pred = rf.predict(X_test)
def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))
RMSE = rmse(y_test, y_pred)
PCC = round(np.corrcoef(y_test, y_pred)[0,1],2)

# Create a DataFrame containing the gene names and predicted values
predictions_df = pd.DataFrame({'Gene Name': gene_names[test_indices], 'Predicted Value': y_pred, 'Original value': Y_target[test_indices], 'PCC': PCC, 'mse': RMSE})

# Save the dataframe to a csv file
predictions_df.to_csv('predicted_values_rf.csv', index=False)

# Using the feature importance attribute of the random forest regressor to get the most important features

# Getting the feature importance
feature_importance = rf.feature_importances_
print(len(feature_importance))

# Sorting the feature importance
sorted_idx = np.argsort(feature_importance)[::-1]

# Getting the names of the features
feature_names = one_hot_df.columns[1:]

# create a dataframe with feature names and importance values
feat_imp_df = pd.DataFrame({'feature_names': feature_names[sorted_idx], 
                            'importance': feature_importance[sorted_idx]})

# save the dataframe to a csv file
feat_imp_df.to_csv('feature_importance_rf.csv', index=False)

#fig = plt.figure(dpi = 150)

# Plotting the feature importance
#plt.bar(feature_names[sorted_idx[:10]],feature_importance[sorted_idx[:10]])
#plt.xlabel('Features')
#plt.xticks(rotation=90)
#plt.ylabel('Importance')

# Save the figure as a PNG file
#fig.savefig('feature_importance_rf.png', dpi=150)
