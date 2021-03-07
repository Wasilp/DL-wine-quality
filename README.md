# DL-wine-quality
Deep learning project realised during Becode training

# Objectives
Get better understanding in pytorch and how neural network work.

### Project overview

The datset was already downloaded and merged for us -> ./data/wine.csv  
The **baseline model** -> **experiments_wineQ.ipynb**  
The **baseline model** was composed with 
- hidden neurons: 100
- hidden layer: 1 
- learning rate: 0.001
- activation function: SiLU
- Last layer activation function: Softmax
- criterion: MSELoss
- optimizer: Adam
- batch size: 50
- epochs: 100

The accuracy for the **baseline model** was first around 50% before upsampling then the accuracy improved to 79.56% on the training set and 77.50% and the test set.

Ray tune optimisation -> **experiments_personal_model_ray.ipynb**

In that notebook i experienced parameters optimisation with Ray tune then i set this parameters in experiments_personal_model.ipynb.
As i couldn't test parameters in the range i wanted (as it take a certain time) this might not be the best parameters but this allowed me to get some experience with ray tune and have a model how will be able to get best params for my futur projects.    

Tuned baseline model -> **experiments_personal_model.ipynb**  
Is the Baseline model that i reworked to be more generic and changed my model parameters from ray tunes best params as mentioned. I also upsampled my target as there was some outliers in the dataset.

- hidden neurons: 200
- hidden layer: 2 
- learning rate: 0.001
- activation function: CELU,SiLU,Sigmoid
- Last layer activation function: Softmax
- criterion: MSELoss
- optimizer: Adam
- batch size: 200
- epochs: 200

The accuracy for this model was a bit higher and in some test reached nearly 95%. The final test i did give me 88.75% for the train set and 85% for the test set.

# Others experiments  
 Standarise dataset which did not change that much in the accuracy -> **experiments_personal_model_stdScaler.ipynb

I have test also others models that i found on kaggle or Medium post  
-> **experiments_wine_should_have_been_simple_way.ipynb**  
->**wine_model_kaggle.py**  

# Pending things to do
As this project was mainly focus on geting our hands dirty with pytorch I didn't dive much in the data this could have increase our accuracy a bit more. 

07/03/2021 **Pierre Wasilewsk**
