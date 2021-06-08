# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:25:59 2021

@author: blgnm
"""

from guassian_processing import *
from get_ztf_data import *
from feature_extraction import *
from classification import *
class LightCurve:
    def __init__(self,ztf_id = None,light_curve_data=None):
        self.ztf_id = ztf_id
        self.data = light_curve_data
        self.gaussian_data = None
        self.alerce_features = None
        self.features = None
        self.pca = None
        self.class_labels = None
    def Data(self):
        self.data = pull_from_antares(self.ztf_id)
    
    def gaussian_processed_regression(self,maximum_sn_length = None, n_samples = 100 ,classification = True):
        self.gaussian_data = gaussian_regression(self.data)
    
    def get_alerce_features(self):
        #Note to self: Go back and fix bugs. Make it work with any alerce feature, and fix bug causing events
        #not on alerce to give an error.
        self.alerce_features = GetAlerce(self.ztf_id)
    
    def get_features(self,train_set=False, n_components = 10):
        Function_Parameters, Coefficients = Wavelet(self.gaussian_data)
        pca = PCA(n_components = n_components, whiten = True)
        if train_set == True:
            Coefficients = pca.fit_transform(pd.DataFrame(Coefficients).dropna())
        self.pca = pca
        self.features = pd.concat([Function_Parameters.reset_index(drop=True),pd.DataFrame(Coefficients).reset_index(drop=True)],axis=1)
    
    def fit_classifier(self):
        features = self.features.iloc[:,1:]
        labels = self.class_labels
        classifier = Classify_SN(features, labels)
    def predict(self):
        return None
    
#%%
x = LightCurve(['ZTF21aatisro','ZTF20abkljlp'])
x.Data()
x.gaussian_processed_regression()
x.get_alerce_features()
x.get_features()

#%%
test = LightCurve()
test.data = SupernovaTrainingData
test.gaussian_processed_regression()
test.get_features(train_set=True)
#%%
print(test.features)
#%%
print(test.data['event'].unique()[167])
#%%
x = test.data
print(x[x['event']=='ZTF18abobkii'])
#%%
pca = PCA(n_components=10)
print(pca.fit_transform(test.features.iloc[500:1000,100:]))
#%%
print(test.features)