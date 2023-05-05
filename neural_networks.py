#!/usr/bin/env python

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
import pytz
from datetime import datetime
from sklearn.model_selection import KFold

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from ast import literal_eval

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import os

# Read the regions -----------
CircadianRegions = 'CircadianRegions_2kb.csv'
CircadianRegions = pd.read_csv(CircadianRegions)

#********************************************************
#
#           TESTING ONLY!!! (comment!)
#
#CircadianRegions = CircadianRegions.sample(frac=0.01, replace=False, random_state=5461) # Sample a fraction of the df
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

class JointDataSet():
    
    """
    Link together a feature matrix and a target array
    """
    
    def __init__(self, X, y):
        """
        Args:
            X (2d np array) - columns are features, rows are samples
            y (1d np array) - values are the target property for each sample
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        """
        len(JointdataSet) returns len(X)
        """
        return len(self.X)
    
    def __getitem__(self, idx):
        """
        Slicing JointDataSet returns (X[sample], y[sample])
        """
        return self.X[idx], self.y[idx]

#********************************************************

def load_data(X, y, val_size=0.2, random_n=None, batch_size=1, shuffle=True, drop_last=True):
    """
    Args:
        X (2d np array) - columns are features, rows are samples
        y (1d np array) - values are the target property for each sample
        val_size (float) - fraction of data to reserve for validation
        random_n (int or None) - if int, use only every random_nth sample from X and y (if None, use all of X and y)
        batch_size (int) - size of each batch for training
        shuffle (bool) - shuffle samples in X, y
        
    Returns:
        {'train' : PyTorch DataLoader object for training set, split into batches of size batch_size,
         'val' : PyTorch DataLoader object for validation set, held in one single batch}
    
    """
   
    # split our training set into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_size, random_state=44)
    
    # slice our training set for de-bugging
    if random_n:
        X_train = X_train[::random_n, :]
        y_train = y_train[::random_n]
    
    # link X and y for training
    joint_dataset_train = JointDataSet(X_train, y_train)
    
    # link X and y for validation
    joint_dataset_val = JointDataSet(X_val, y_val)
    
    # return the "loaded" data
    return {'train' : DataLoader(dataset=joint_dataset_train, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last),
            'val' : DataLoader(dataset=joint_dataset_val, batch_size = len(joint_dataset_val), shuffle=shuffle, drop_last=drop_last)}

#********************************************************

def construct_model(X_train, hidden_layer_sizes, output_size, act_fn, dropout, norm):

    input_size = X_train.shape[1] # how many features go into the first layer

    # initialize a sequence of layers
    model = nn.Sequential()

    # add layers
    ## general flow: linear + activation --> linear + activation --> ...
    for i in range(len(hidden_layer_sizes)):
        
        if norm:
            model.add_module('norm%s' % str(i+1), nn.BatchNorm1d(input_size if i == 0 else hidden_layer_sizes[i-1]))
        
        if i == 0:
            model.add_module('fc%s' % str(i+1), nn.Linear(input_size, hidden_layer_sizes[i]))
            model.add_module('relu1', act_fn)              
        else:
            model.add_module('fc%s' % str(i+1), nn.Linear(hidden_layer_sizes[i-1], hidden_layer_sizes[i]))
            model.add_module('relu%s' % str(i+1), act_fn)
            
        if dropout:
            model.add_module('drop%s' % str(i+1), nn.Dropout1d(dropout))

    # add our last layer (linear + softmax to get probabilities)
    model.add_module('fc_last', nn.Linear(hidden_layer_sizes[i], output_size))
    model.add_module('soft', nn.Softmax())    

    # print the model
    print(model)
    
    return model

#********************************************************

def rmse(actual, pred):

    #pred = torch.argmax(pred)
    
    #print(actual.shape)
    #print(pred.shape)
    
    actual = actual.detach().numpy()
    pred = pred.detach().numpy()
    
    return np.sqrt(np.mean((actual - pred)**2))

def score(actual, pred, metric='rmse'):
    
    if metric == 'rmse':
        return rmse(actual, pred)

#********************************************************
    
def train_one_batch(x_batch, y_batch, X_val, y_val, model, loss_fn, optimizer, metric):
    """
    Args:
        x_batch (np.array) : feature matrix for training batch
        y_batch (np.array) : target array for training batch
        X_val (np.array) : feature matrix for validation
        y_val (np.array) : target array for validation
        model (torch.nn.Module) : neural network
        loss_fn (torch.nn.LossFunction) : loss function for optimization
        optimizer (torch.nn.Optimizer) : optimizer for training
        
    Returns:
        {'rmse' : rmse on training batch,
         'loss' : loss on training batch,
         'model' : model trained on batch,
         'optimizer' : optimizer after running on batch}
    
    """
    
    # make our predictions on this batch of training data
    train_pred = model(x_batch)
    
    # update our loss function based on these predictions
    if metric in ['accuracy', 'f1']:
        #train_pred = train_pred.type(torch.LongTensor)
        y_batch = y_batch.type(torch.LongTensor)
    elif metric in ['rmse', 'mae']:
        train_pred = torch.flatten(train_pred)
        y_batch = torch.flatten(y_batch)
            
    loss = loss_fn(train_pred, y_batch)
    
    # propagate the loss backward
    loss.backward()
    
    # step our optimizer forward
    optimizer.step()
    
    # return our optimizer to stage 0
    optimizer.zero_grad()
    
    # compute the rmse
    
    train_rmse = score(y_batch, train_pred, metric=metric)
    
    # compute the value for the loss
    train_loss = loss.item()
    
    if score in ['accuracy', 'f1']:
        train_loss = train_loss*y_batch.size(0)
    
    # make predictions on the validation set
    val_pred = model(X_val)
    if metric in ['accuracy', 'f1']:
        y_val = y_val.type(torch.LongTensor)
    
    # compute rmse and loss on validation set
    val_rmse = score(y_val, val_pred, metric=metric)
    
    val_loss = loss_fn(val_pred, y_val).item()
    if score in ['accuracy', 'f1']:
        val_loss = val_loss*y_val.size(0) / len(y_val) * y_batch.size(0)
        
    # also return our model and optimizer! to use them for the next batch
    return {'rmse' : {'train' : train_rmse,
                     'val' : val_rmse},
            'loss' : {'train' : train_loss,
                      'val' : val_loss},
            'model' : model,
            'optimizer' : optimizer}

#********************************************************
def train_epochs(num_epochs, data_loader, model, loss_fn, optimizer, metric='rmse', verbose=0):
    """
    Args:
        num_epochs (int) : number of epochs to train for
        data_loader (torch.datasets.DataLoader) : data_loader containing X, y in batches
        model (torch.nn.Module) : neural network
        loss_fn (torch.nn.LossFunction) : loss function
        optimizer (torch.nn.Optimizer) : optimizer
        
    Returns:
        {'scores' : 
            {epoch (int) : 
                {'acc' or 'loss' : 
                    {'train' or 'val' :
                        mean score for that batch}
         'model' : model after training,
         'optimizer' : optimizer after training}
    
    """
    # need our model, loss function, and optimizer to start somewhere
    initial_model = {'model' : model,
                     'loss_fn' : loss_fn,
                     'optimizer' : optimizer}    
    
    # specify sources for data to assess
    data_sources = ['train', 'val']
    
    # specify metrics to assess model with
    outputs = ['rmse', 'loss']
    
    # initialize a dictionary to store metrics in
    scores = {epoch : 
              {output : 
                       {src : [] 
                        for src in data_sources} 
               for output in outputs} 
              for epoch in range(num_epochs)}
    
    # get our validation data (the same for each batch in every epoch)
    for X_val, y_val in data_loader['val']:
        print('')
        
    # loop through epochs (training steps)
        # note: one epoch means we've gone through all batches in our training data
    for epoch in range(num_epochs):
        if verbose > 0:
            print('\n epoch %i' % (epoch+1))
        batch_count = -1        
        
        # loop through all our batches in each epoch
        for x_batch, y_batch in data_loader['train']:
            batch_count += 1
            
            # if we're on our first batch and first epoch, use our initial model
            if not batch_count and not epoch:
                results = initial_model.copy()
            
            # otherwise, use the model we optimized on the most recent batch
            model = results['model']
            optimizer = results['optimizer']
            
            # update the model, optimizer with current batch
                        
            results = train_one_batch(x_batch, y_batch, X_val, y_val, model, loss_fn, optimizer, metric)
        
            # store the results on this batch
            for output in outputs:
                for src in data_sources:
                    scores[epoch][output][src].append(results[output][src])
        
        # aggregate the results over all batches for the current epoch
        for output in outputs:
            for src in data_sources:
                score = np.mean(scores[epoch][output][src])
                if verbose > 1:
                    print('%s (%s) : %.3f' % (output, src, score))
                scores[epoch][output][src] = score
    
    # return the results, the final model, and the final optimizer
    return {'scores' : scores,
            'model' : results['model'],
            'optimizer' : results['optimizer']}

#********************************************************
#Function that combines all the above functions into one

def do_it_all(X_train, y_train, random_n, hidden_layer_sizes, num_epochs,learning_rate,
              batch_size, val_size=0.1,
              output_size=1, act_fn=nn.LeakyReLU(), dropout=0.5, norm='batch',
              loss_fn=nn.MSELoss(),
              l2_penalty=0.8, metric='rmse', verbose=0,
              seed=44
             ):
    """
    Args:
        0. Random seeding:
            seed (int) : random seed
            
        1. Data loading:
            X_train (2d np array) : feature matrix
            y_train (1d np array) : target array
            val_size (float) : fraction of training to reserve for validation
            random_n (int) : 
                - if random_n = 100, use only every 100th training point (for debugging)
                - set random_n = 1 to use all of your data
            batch_size (int) : number of training points to use for each parameter update step
            
        2. Model construction:
            hidden_layer_sizes (list) : list of neurons in each hidden layer
                - [256] would be one hidden layer with 256 neurons
                - [256, 128] would be a hidden layer with 256 neurons followed by a hidden layer with 128 neurons
            output_size (int) : number of classes in classification problem or 1 for regression problem
            act_fn (torch.nn.Module) : activation function after each linear hidden layer
            dropout (float or None) : if None, all hidden layers are fully connected, else dropout fraction of connections are dropped
            norm (str or None): if None, no normalization, else use norm to normalize before each layer
        
        3. Loss function:
            loss_fn (torch.nn.Module) : loss function for training
        
        4. Optimizer:
            learning_rate (float) : learning rate for parameter updates
            l2_penalty (float) : magnitude of L2 regularization
            
        5. Training:
            num_epochs (int) : how many epochs to train for
            metric (str) : what to "score" in addition to loss
            verbose (int) : how much to print
            
        6. Visualization:
            baseline (float) : value to plot a horizontal line on as sa baseline for comparison
            problem (str) : 'classification' or 'regression'
            
    Returns:
        dictionary with trained model and scores
    """
    
    print('\nusing %i (%.2f) data points' % (len(y_train)/random_n, 1/random_n))
    torch.manual_seed(seed)
    
    data_loader = load_data(X_train, y_train, val_size=val_size, random_n=random_n, batch_size=batch_size)
    
    model = construct_model(X_train, hidden_layer_sizes, output_size, act_fn, dropout, norm)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_penalty)
    
    results = train_epochs(num_epochs, data_loader, model, loss_fn, optimizer, metric=metric, verbose=verbose)

    data_sources = ['train', 'val']
    outputs = ['rmse', 'loss']
    scores = results['scores']

    plot_data = {src : 
                 {output : 
                  [scores[epoch][output][src] for epoch in scores] 
                  for output in outputs} for 
                 src in data_sources}
    
    print('best scores:\n')
    print('  training = %.2f' % max(plot_data['train']['rmse']))
    print('  validation = %.2f' % max(plot_data['val']['rmse']))

    return results

#********************************************************

print("Getting Results")


#Getting Results
results_nn = do_it_all(X_train, y_train, 
                    random_n=2, hidden_layer_sizes=[256,256,256], num_epochs=100, 
                    learning_rate= 10**-5, batch_size=50)

print("Results gotten")
#Saving dataframe
results_nn_df = pd.DataFrame(results_nn['scores'])
results_nn_df.to_csv('results__nn_df_5.csv')

model = results_nn["model"]
X_test_torch = torch.from_numpy(X_test).float()
y_pred = model(X_test_torch)

y_pred = y_pred.detach().numpy()

def rmse_array(y, y_pred):
    return np.sqrt(np.mean((y - y_pred)**2))

RMSE = rmse_array(y_test, y_pred)

y_pred = np.squeeze(y_pred)


predictions_df = pd.DataFrame({'Gene Name': gene_names[test_indices], 'Predicted Value': y_pred, 'Original value': Y_target[test_indices], 'rmse': RMSE})
predictions_df.to_csv('predicted_values_nn5.csv', index=False)

print("Batch of 50,used 256 neurons one layer, learning rate 10-5, batch size 10,l2 penalty 0.75")
#********************************************************

