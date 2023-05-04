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

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler




# Read the regions -----------

CircadianRegions = '../output/CircadianRegions_2kb.csv'
CircadianRegions = pd.read_csv(CircadianRegions)



#********************************************************
#
#           TESTING ONLY!!! (comment!)
#
CircadianRegions = CircadianRegions.sample(frac=0.1, replace=False, random_state=5461) # Sample a fraction of the df
#
#********************************************************



# Prepare data -----------
data_df = CircadianRegions[['gene','JTK_adjphase','region2kb']] # keep relevant cols
data_df = data_df.sort_values('gene') # make sure df is sorted by gene
# CircadianRegions = None




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


# One hot encoding -----------

# Create features df
features_df = pd.concat([data_df['gene'], Features], axis=1)
# data_df = None
# Set and fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
one_hot = mlb.fit_transform(features_df["features"]) # fit
# Create a new df with the one-hot encoding and the gene column
one_hot_df = pd.DataFrame(one_hot, columns=mlb.classes_)
one_hot_df["gene"] = features_df["gene"].values
one_hot_df = one_hot_df[["gene"] + list(mlb.classes_)] # "gene" col to front
one_hot_df = one_hot_df.set_index('gene') # Set gene as index
# Remove features containin "N" (only 6)
one_hot_df = one_hot_df.loc[:, ~one_hot_df.columns.str.contains('N')]

# Convert to numpy arrays
feature_names = np.array(one_hot_df.columns) # get feature names
gene_names = np.array(one_hot_df.index) # get gene names
np_data = one_hot_df.values  # Convert data to numpy array

print('computed one-hot-encoding')



# Split data for training -----------
# Appears to be faster with a df than arrays

# Data for models
X_features = one_hot_df
Y_target = data_df["JTK_adjphase"]
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_features, Y_target, test_size=0.75, random_state=5461)

print('Finished splitting data')




# Select and train models -----------

# Create empty dfs to output
ModelsPerformance = pd.DataFrame()
FeatureImportance = pd.DataFrame()

# Create models
# knn = KNeighborsRegressor(n_neighbors=5)
# svr = SVR(kernel='rbf', C=1, gamma=0.1)
rfr = RandomForestRegressor(n_estimators=1, random_state=0)
# gbr = GradientBoostingRegressor(n_estimators=1, learning_rate=0.1, max_depth=1, random_state=0, loss='ls')
# Store them in dictionary
# Models = {'rfr':rfr,
#           'svr':svr,
#           'gbr':gbr,
#           'knn':knn}
Models = {'rfr':rfr}

# Select model
model_name = next(iter(Models)) # Name
model = next(iter(Models.values())) # Model

# Train the algorithm on the training set
start_time = time.time() # ***** Start timer *****
model.fit(X_train, y_train)

# Use the trained algorithm to predict the target variable of the test set
y_pred = model.predict(X_test)

# Feature importance
# if model_name == 'rfr' | model_name == 'gbr':
feature_importance = model.feature_importances_
# Create feature importance df for model
colname =  model_name + '_Importance'
FI_df = pd.DataFrame({'Feature': one_hot_df.columns, colname: feature_importance})
FI_df = FI_df.sort_values(by=[colname], ascending=False)
# Append to greater df
FeatureImportance = pd.concat([FeatureImportance,FI_df], axis=0)


end_time = time.time() # ***** End timer *****
# Time to run
execution_time = round(end_time - start_time, 4)
print('Finished model ',model_name)



# Performance -----------

# Get performance metrics into a df
mse = mean_squared_error(y_test, y_pred)
PCC = round(np.corrcoef(y_test, y_pred)[0,1],2)
performance_df = pd.DataFrame({'Model':model_name,
                               'PCC':PCC,
                               'MSE':mse,
                               'ExecTime':execution_time},index=[0])
# Append to performance df
ModelsPerformance = pd.concat([ModelsPerformance,performance_df], axis=0)




# Export data -----------

tz = pytz.timezone('US/Central') # set timezone (tz) to local
now = datetime.now(tz).strftime('%Y%m%d_%H%M') # Get current time
# Filenames
features_csv = '../output/ML_features_'+now+'.csv'
performance_csv = '../output/ML_performance_'+now+'.csv'
# Export csv
ModelsPerformance.to_csv(performance_csv)
FeatureImportance.to_csv(features_csv)
