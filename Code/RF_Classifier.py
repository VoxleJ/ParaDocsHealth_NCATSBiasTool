# -*- coding: utf-8 -*-
"""
@author: vjha1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

preprocessedData = pd.read_csv('Labeled/KA_btwn_2_5_Labeled_Combined_PreProcessed.csv') #takes an already preprocessed file
#prior preprocessing to keep interpretability

# preprocessedData = preprocessedData.reset_index()

#%%
y = preprocessedData.pop('label')
X = preprocessedData.drop('id', axis=1)

seed = 600
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

X, y = oversample.fit_resample(X,y) #to mitigate class imbalance issues

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state = seed) #80/20 split

from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(min_samples_leaf=10, n_estimators=150, bootstrap=True, oob_score=True,
    random_state = seed)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(rf_classifier)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

#%%
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
acc = accuracy_score(y_test, y_pred)

print('ACC:', acc)


train_probs = pipe.predict(X_train) 
probs = pipe.predict_proba(X_test)[:,1]

train_predictions = pipe.predict(X_train)

train_AUC = roc_auc_score(y_train, train_probs)
test_AUC = roc_auc_score(y_test, probs)
#code augmented from scikitlearn documentation and Joe Tran, https://towardsdatascience.com/my-random-forest-classifier-cheat-sheet-in-python-fedb84f8cf4f

def evaluate_model(y_pred, probs,train_predictions, train_probs):    
    baseline = {}    
    baseline['recall']=recall_score(y_test, [1 for _ in range(len(y_test))])
    baseline['precision'] = precision_score(y_test,[1 for _ in range(len(y_test))])    
    baseline['roc'] = 0.5    
    results = {}    
    results['recall'] = recall_score(y_test, y_pred)    
    results['precision'] = precision_score(y_test, y_pred)    
    results['roc'] = roc_auc_score(y_test, probs)    
    train_results = {}    
    train_results['recall'] = recall_score(y_train, train_predictions)    
    train_results['precision'] = precision_score(y_train, train_predictions)    
    train_results['roc'] = roc_auc_score(y_train, train_probs)    
    for metric in ['recall', 'precision', 'roc']:  
        print('Recall',round(baseline[metric], 2)) 
        print('Precision', round(results[metric], 2))
        print('ROC', round(train_results[metric], 2))
                 
                 # Calculate false positive rates and true positive rates    
    base_fpr, base_tpr, _ = roc_curve(y_test, [1 for _ in range(len(y_test))])    
    model_fpr, model_tpr, _ = roc_curve(y_test, probs)    
    plt.figure(figsize = (8, 6))
    plt.rcParams['font.size'] = 16    # Plot both curves    
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate'); plt.title('ROC Curves');
    plt.show();
    

#get metrics and plot ROC/AUC curve
evaluate_model(y_pred,probs,train_predictions,train_probs)


confusion_matrix(y_test, y_pred)


#%%

from sklearn.model_selection import RandomizedSearchCV
#code augmented from scikitlearn documentation and Joe Tran, https://towardsdatascience.com/my-random-forest-classifier-cheat-sheet-in-python-fedb84f8cf4f
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 700, num = 50)]

max_features = ['sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 4, 10]
bootstrap = [True, False] 

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_leaf_nodes': [None] + list(np.linspace(10, 50, 500).astype(int)),
               'bootstrap': bootstrap}


rf = RandomForestClassifier(oob_score=True)

rf_tune = RandomizedSearchCV(                        
    estimator = rf,                        
    param_distributions = random_grid,                        
    n_iter = 50, cv = 5,                        
    verbose=2, random_state=seed,                         
    scoring='roc_auc')

pipe_tune = make_pipeline(rf_tune)
pipe_tune.fit(X_train, y_train)

print(rf_tune.best_params_)

#%% view best model and calculate metrics
bestm = rf_tune.best_estimator_

pipe_best = make_pipeline(bestm)

pipe_best.fit(X_train, y_train)

y_pred_best = pipe_best.predict(X_test)

train_rf_pred = pipe_best.predict(X_train)

train_rf_probs = pipe_best.predict_proba(X_train)[:,1]
rf_probs = pipe_best.predict_proba(X_test)[:,1]

evaluate_model(y_pred_best, rf_probs, train_rf_pred, train_rf_probs)

from sklearn.metrics import ConfusionMatrixDisplay

confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()
cm_disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_best)).plot()

Best_acc = accuracy_score(y_test, y_pred_best)

print('Best ACC:', Best_acc)

#%% save model
import os
import joblib

joblib.dump(bestm, "./rf_best_noteBias.joblib")