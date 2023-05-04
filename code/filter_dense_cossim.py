#!/usr/bin/env python

# Code to filter features based on density and PCA

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


# start timer
start_time = time.time()


# Read the regions -----------
CircadianRegions = '../output/CircadianRegions_2kb.csv'
CircadianRegions = pd.read_csv(CircadianRegions)

#********************************************************
#
#           TESTING ONLY!!! (comment!)
#
# CircadianRegions = CircadianRegions.sample(frac=0.5, replace=False, random_state=5461) # Sample a fraction of the df
#
#********************************************************




# Prepare data -----------
data_df = CircadianRegions[['gene','JTK_adjphase','region2kb']] # keep relevant cols
data_df = data_df.sort_values('gene') # make sure df is sorted by gene



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
# Set and fit MultiLabelBinarizer
mlb = MultiLabelBinarizer()
one_hot = mlb.fit_transform(features_df["features"]) # fit
# Create a new df with the one-hot encoding and the gene column
one_hot_df = pd.DataFrame(one_hot, columns=mlb.classes_)
one_hot_df["gene"] = features_df["gene"].values
one_hot_df = one_hot_df[["gene"] + list(mlb.classes_)] # "gene" col to front
one_hot_df = one_hot_df.set_index('gene') # Set gene as index
feature_names = np.array(one_hot_df.columns) # get feature names
print('computed one-hot-encoding')


# Filter based on density -----------

# Copy data
df = one_hot_df
# Remove features containin "N" (only 6)
df = df.loc[:, ~df.columns.str.contains('N')]

# Get densities
feature_densities = df.sum() / df.shape[0]
feature_densities = feature_densities.sort_values(ascending=False)

# Filter out bottom and top thresholds (quantile-based)
# Calculate the nth percentile values
density_percentiles = np.percentile(feature_densities, [0,25,50,75,100])
print('Percentiles 0,25,50,75,100 are: \n', str(density_percentiles))
bottom_thresh = density_percentiles[1]
top_thresh = density_percentiles[2]
#filter
filtered_series = feature_densities[
    (feature_densities > bottom_thresh) & (feature_densities < top_thresh)]
filtered_features = filtered_series.index.to_list() # list names
filtered_features = df[filtered_features] # subset
print('Dimensions after density-based filter ',str(filtered_features.shape))


# Filter based on cosine similarity -----------

# Copy df
df = filtered_features
print(str(df.shape))

# Initialize a list to store the selected feature indices
all_features = df.columns.to_list()
selected_features = [all_features[0]]

# Get last selected feature (Feature_X)
Feature_X = df[all_features[0]].astype(bool).to_numpy().reshape(1, -1) #bool
Feature_X = Feature_X.astype(int) # Needs to be a binary array

all_features.pop(0) # Remove top feature
similarities = []

# Loop until all features have been selected or rejected
while len(all_features) > 0:

    # Get last selected feature (Feature_X)
    Feature_Y = df[all_features[0]].astype(bool).to_numpy().reshape(1, -1) #bool
    Feature_Y = Feature_Y.astype(int) # Needs to be a binary array

    # Compute cosine similarity
    cos_similarity = cosine_similarity(Feature_X, Feature_Y)
    cos_similarity = cos_similarity[0][0]
    similarities.append(cos_similarity)

    if cos_similarity > 0.1:
        print('Cosinse similarity = ', cos_similarity)
        print('Selected features: ',str(len(selected_features)), 'Remaining: ', str(len(all_features)))
    # Accept feature is absolute value is greater than 75%
    if abs(cos_similarity) < 0.75:
        selected_features.append(all_features[0])
        # print(str(selected_features))
    # Drop next top col
    all_features.pop(0)

# filter
unique_features = np.unique(selected_features)
n_unique_features = len(unique_features) # get number of features
# Filtered dataset
filtered_df = one_hot_df.loc[:, unique_features]
# Export
filename = '../output/FilteredFeatures_CosSim_n' + str(n_unique_features) + '.csv'
filtered_df.to_csv(filename)


# end timer
end_time = time.time()
# calculate elapsed time
elapsed_time = end_time - start_time

print("Elapsed time: ", round(elapsed_time,2), " seconds")
