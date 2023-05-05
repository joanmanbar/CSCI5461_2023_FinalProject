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

from sklearn.model_selection import KFold


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

# Function that performs the cross validation
def run_cv(n_folds, model, X_train, y_train):
    """
    Args:
        n_folds (int) : how many folds of CV to do
        model (sklearn Model) : what model do we want to fit
        X_train (np.array) : feature matrix
        y_train (np.array) : target array
        
    Returns:
        a dictionary with scores from each fold for training and validation
            {'train' : [list of training scores],
             'val' : [list of validation scores]}
            - the length of each list = n_folds
    """
    
    folds = KFold(n_splits=n_folds).split(X_train, y_train)

    train_scores, val_scores = [], []
    for k, (train, val) in enumerate(folds):
        
        X_train_cv = X_train[train]
        y_train_cv = y_train[train]

        X_val_cv = X_train[val]
        y_val_cv = y_train[val]

        model.fit(X_train_cv, y_train_cv)

        y_train_cv_pred = model.predict(X_train_cv)
        y_val_cv_pred = model.predict(X_val_cv)

        train_acc = rmse(y_train_cv, y_train_cv_pred)
        val_acc = rmse(y_val_cv, y_val_cv_pred)

        train_scores.append(train_acc)
        val_scores.append(val_acc)

    print('%i Folds' % n_folds)
    print('Mean training rmse = %.3f +/- %.4f' % (np.mean(train_scores), np.std(train_scores)))
    print('Mean validation rmse = %.3f +/- %.4f' % (np.mean(val_scores), np.std(val_scores)))
    
    return {'train' : train_scores,
            'val' : val_scores}

def rmse(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

# Split data for training -----------

# Data for models
X_features = one_hot_df.iloc[:, 2:].values
Y_target = data_df["JTK_adjphase"].values
gene_names = data_df["gene"].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_features, Y_target, range(len(Y_target)), test_size=0.2, random_state=5461)

# Random Forest Model
xgb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, verbose=3, random_state=5461)

n_folds = 3
print("before Cross Validation")
cv = run_cv(n_folds, xgb, X_train, y_train)
print("Cross Validation done")
print("\nFor estimators=100, max depth = 10 and learning rate of .1:")
print(cv)
cross_val_df = pd.DataFrame(cv)
cross_val_df.to_csv('cross_val_xgb.csv', index=False)

# Predicting the values for the test set
y_pred = xgb.predict(X_test)

RMSE = rmse(y_test, y_pred)
PCC = round(np.corrcoef(y_test, y_pred)[0,1],2)

# Create a DataFrame containing the gene names and predicted values
predictions_df = pd.DataFrame({'Gene Name': gene_names[test_indices], 'Predicted Value': y_pred, 'Original value': Y_target[test_indices], 'PCC': PCC, 'rmse': RMSE})

# Save the dataframe to a csv file
predictions_df.to_csv('predicted_values_xgb_cv.csv', index=False)

# Using the feature importance attribute of the random forest regressor to get the most important features

# Getting the feature importance
feature_importance = xgb.feature_importances_
print(len(feature_importance))

# Sorting the feature importance
sorted_idx = np.argsort(feature_importance)[::-1]

# Getting the names of the features
feature_names = one_hot_df.columns[1:]

# create a dataframe with feature names and importance values
feat_imp_df = pd.DataFrame({'feature_names': feature_names[sorted_idx], 
                            'importance': feature_importance[sorted_idx]})

# save the dataframe to a csv file
feat_imp_df.to_csv('feature_importance_xgb_cv.csv', index=False)

#fig = plt.figure(dpi = 150)

# Plotting the feature importance
#plt.bar(feature_names[sorted_idx[:10]],feature_importance[sorted_idx[:10]])
#plt.xlabel('Features')
#plt.xticks(rotation=90)
#plt.ylabel('Importance')

# Save the figure as a PNG file
#fig.savefig('feature_importance_xgb.png', dpi=150)
