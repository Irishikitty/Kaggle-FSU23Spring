import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import seaborn as sns
import json 

import tensorflow as tf
import tensorflow_decision_forests as tfdf
import matplotlib.pyplot as plt

# pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')

# read data
train = pd.read_csv('training_data.csv')
test = pd.read_csv('test_data.csv')

# transformation
train['SalePrice'] = train['SalePrice'].transform(lambda x:np.log(x))
train['RecordingDate'] = pd.to_datetime(train['RecordingDate'])
train['RecordMonth'] = train['RecordingDate'].map(lambda x: x.month).astype('float64')
train['RecordYear'] = train['RecordingDate'].map(lambda x: x.year).astype('float64')
train = train.drop('RecordID', axis=1)
train = train.drop('RecordingDate', axis=1)

# select var - read valid variables after handling missing values
with open('variables.txt','r') as f:
    vars = f.readlines()
vars = [i.strip() for i in vars]

exclude_lst = []
label = 'SalePrice'
for var in vars: # train.columns
    try:
        tfdf.keras.pd_dataframe_to_tf_dataset(train[[var,'SalePrice']], label=label, task = tfdf.keras.Task.REGRESSION)
    except:
        exclude_lst.append(var)
        print(var)
        
exclude_lst.remove('SalePrice')

train_ds_pd = train[vars]
label = 'SalePrice'
train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)

# ========================================================
# modelling

print('-'*20)
print('start model training')
print('-'*20)
tuner = tfdf.tuner.RandomSearch(num_trials=50)

tuner.choice("num_trees", [300,400,500])
tuner.choice("min_examples", [6,7,8,9])
local_search_space = tuner.choice("growing_strategy", ["LOCAL"])
local_search_space.choice("max_depth", [8,10,12,15,20])
global_search_space = tuner.choice("growing_strategy", ["BEST_FIRST_GLOBAL"], merge=True)
global_search_space.choice("max_num_nodes", [16, 32, 64, 128, 256])
tuner.choice("use_hessian_gain", [True, False])
tuner.choice("shrinkage", [0.15,0.2,0.25])
tuner.choice("num_candidate_attributes_ratio", [0.4, 0.5, 0.6, 0.7])

GBT = tfdf.keras.GradientBoostedTreesModel(task = tfdf.keras.Task.REGRESSION, tuner=tuner)
GBT.fit(train_ds, verbose=2)

tuning_logs = GBT.make_inspector().tuning_logs()
tuning_logs.to_csv('gbt.csv')

print('Finished.')