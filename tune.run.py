import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.utils import resample
from torch.utils.data import DataLoader, TensorDataset, random_split
import tempfile
from ray import tune
from functools import partial
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
import ray
from ray.tune.schedulers import ASHAScheduler
#from experiments_ray_tune import Network


##### NEED TO MOVE THIS SECTION TO OTHER FILE ######################
filename = "./data/wine.csv"
df = pd.read_csv(filename)
# drop col index
df = df.drop(['index'],axis = 1)

df_majority = df[df['quality']== 6]
for i in range(3,10):
    majority_len = df[df['quality']== 6].shape[0]
    if i != 6:
        minority_len = df[df['quality'] == i].shape[0]

        df_minority = df[df['quality'] == i]

        df_majority_upsampled = resample(df_minority,replace=True,n_samples = majority_len,random_state=1)

        df_majority = df_majority.append(df_majority_upsampled)
        

df = df_majority
df = df.sample(frac=1).reset_index(drop=True) # Shuffle dataframe

train_test_split_fraction = 1
split_index = int(df.shape[0] * train_test_split_fraction)
df_train = df[:split_index]
df_test = df[split_index:]

target = df['quality'].to_numpy()
target = target.reshape(19852,1) # One hot encode

X_train = df_train.drop('quality', axis = 1).values
X_test = df_test.drop('quality', axis = 1).values

y_train = target[:split_index]
y_test = target[split_index:]

dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).float())

########################################################################################

