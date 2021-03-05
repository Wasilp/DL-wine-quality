import pandas as pd
import numpy as np
import torchvision
import torch.nn as nn
from torchvision.datasets.utils import download_url
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

"""[summary]
https://www.kaggle.com/awadhi123/red-wine-quality-pytorch
Returns:
[type]: [description]
"""
#import of df
filename = "./data/wine.csv"
df = pd.read_csv(filename)
df.sample(frac=1).reset_index(drop=True) # Shuffle dataframe
df = df.drop(['index'],axis = 1) # drop col index


input_cols = list(df.columns)[:-1]
output_cols = ['quality']

def dataframe_to_arrays(df):
    # Make a copy of the original dataframe
    df1 = df.copy(deep=True)
    # Extract input & outupts as numpy arrays
    inputs_array = df1[input_cols].to_numpy()
    targets_array = df1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_array = dataframe_to_arrays(df)

inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)

print(inputs_array.shape,targets_array.shape)


"""
Convert numpy array to torch tensor

"""
inputs = torch.Tensor(inputs_array)
targets = torch.Tensor(targets_array)


dataset = TensorDataset(inputs, targets)


num_rows = len(df)
val_percent = 0.01 # between 0.1 and 0.2
val_size = int(num_rows * val_percent)
train_size = num_rows - val_size



train_df, val_df = random_split(dataset, [train_size, val_size]) 


batch_size = 50
train_loader = DataLoader(train_df, batch_size, shuffle=True)
val_loader = DataLoader(val_df, batch_size)

input_size = len(input_cols)
output_size = len(output_cols)


class WineModel(nn.Module):
    def __init__(self):
        super().__init__()     
        self.linear = nn.Linear(input_size, output_size) # fill this (hint: use input_size & output_size defined above)
        #model initialized with random weight
        
    def forward(self, xb):
        out = self.linear(xb)             # batch wise forwarding
        return out
    
    def training_step(self, batch):
        inputs, targets = batch 
        # Generate predictions
        out = self(inputs)         
        # Calcuate loss
        loss = F.l1_loss(out, targets)  # batch wise training step and loss
        return loss
    
    def validation_step(self, batch):
        inputs, targets = batch
        # Generate predictions
        out = self(inputs)
        # Calculate loss
        loss =F.l1_loss(out, targets)       # batch wise validation and loss    
        return {'val_loss': loss.detach()}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine val losses of all batches as average
        return {'val_loss': epoch_loss.item()}
    
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))

model =  WineModel()
list(model.parameters())

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append(result)  #appends total validation loss of whole validation set epoch wise
    return history

result = evaluate(model,val_loader) # Use the the evaluate function
print(result)



epochs = 1000
lr = 1e-2
history1 = fit(epochs, lr, model, train_loader, val_loader)

epochs = 1000
lr = 1e-4
history3 = fit(epochs, lr, model, train_loader, val_loader)

val_loss = evaluate(model,val_loader)

def predict_single(input, target, model):
    inputs = input.unsqueeze(0) 
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)

input, target = val_df[0]
predict_single(input, target, model)