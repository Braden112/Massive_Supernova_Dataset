# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 15:46:34 2021

@author: blgnm
"""
import pickle
from imblearn.ensemble import BalancedRandomForestClassifier

class Classification:
    def __init__(self, lightcurve):
        self.lightcurve = lightcurve
        self.classifier = pickle.load(open('sn_classifier.sav', 'rb'))
        
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