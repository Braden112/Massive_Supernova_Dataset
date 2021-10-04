#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 20:11:35 2021

@author: bgarrets
"""

from pickle import load
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os


class Classification:
    def __init__(self, lightcurve):
        self.lightcurve = lightcurve
        self.classifier = load(open('SN_classification.sav', 'rb'))
        self.host_ra = None
        self.host_dec = None
   
    def fit_classifier(self):
        Data = self.lightcurve.features.iloc[:,1:]
        labels = self.lightcurve.class_labels
        classifier =  BalancedRandomForestClassifier(criterion = 'entropy', 
                                                      max_features = 'sqrt', 
                                                      n_estimators = 1000, n_jobs = -1,
                                                      max_depth = 15, min_samples_leaf = 1, 
                                                      min_samples_split = 2,replacement = True, 
                                                      class_weight = 'balanced_subsample')
        self.classifier = classifier.fit(Data, np.array(labels).ravel())
     
    def classify_events(self, probability = False):
        
        if probability == True:
            return self.classifier.predict_proba(self.lightcurve.features.iloc[:,1:])
        else:
            return pd.DataFrame(self.classifier.predict(self.lightcurve.features.iloc[:,1:])).replace([0,1,2,3],['SN Ia','SN II', 'SN Ib/c', 'SLSN'])
        
    
        
        
        
        
        
