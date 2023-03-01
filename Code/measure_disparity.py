# -*- coding: utf-8 -*-
"""
@author: vjha1
"""

#Measure Disparity

#takes a preprocessed file as input
import pandas as pd
import FeatureExtraction


filename = input("Enter the preprocessed Data")

preprocessedData = pd.read_csv('Labeled/KA_btwn_2_5_Labeled_Combined_PreProcessed.csv') #takes an already preprocessed file

trial = input('If you would like to extract features type 1:')

if trial == 1:
    FeatureExtraction #loaded with sample file

#%%

#Apply Random Forest Model 

import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import numpy as np

loaded_rf = joblib.load("./rf_best_noteBias.joblib")

#Select single value
    
# testing = preprocessedData.iloc[390].to_numpy()    

y = preprocessedData.pop('label')
X = preprocessedData.drop('id', axis=1)

atestlabels = loaded_rf.predict(X)

#example
allfeatures = pd.read_csv('Labeled/KA_btwn_2_5_Labeled_Combined.csv')

#%% example

print("Example of Biased entry: ", preprocessedData.iloc[388])
print("With all features: ", allfeatures.iloc[388] )

crucial_bias = allfeatures.crucial_statements.iloc[388]
print('Crucial Statements', crucial_bias)

#%% saving dataframe
abc = pd.Series(atestlabels)
abc.name = 'Model_Labels'

TotalOutput = pd.concat([abc, allfeatures], axis=1)

