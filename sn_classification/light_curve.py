# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:25:59 2021

@author: blgnm
"""

from guassian_processing import *
from get_ztf_data import *
from feature_extraction import *
from original_classification import *
from sklearn.decomposition import PCA
import sfdmap

class LightCurve:
    def __init__(self,ztf_id = None,light_curve_data=None):
        self.ztf_id = ztf_id
        self.data = light_curve_data
        self.gaussian_data = None
        self.alerce_features = None
        self.features = None
        self.pca =  pickle.load(open('sn_pca.sav', 'rb'))
        self.class_labels = None
        self.ra = None
        self.dec = None
        self.ebv = None

    def Data(self):
        self.data = pull_from_antares(self.ztf_id)
    
    def get_location(self):
        location = GetRaDec(self.ztf_id).reset_index(drop=True)
        self.ra = location['ra']
        self.dec = location['dec']
    def milky_way_extinction(self):
        dustmap = sfdmap.SFDMap(r"C:/Users/blgnm/Desktop/sfddata-master/")
        ebv = list()
        for i in range(0, len(self.ra)):
            ebv.append(dustmap.ebv(self.ra[i],self.dec[i]))
        self.ebv = ebv
    def gaussian_processed_regression(self,maximum_sn_length = None, n_samples = 100 ,classification = True):
        self.gaussian_data = gaussian_regression(self.data)
    
    def get_alerce_features(self):
        #Note to self: Go back and fix bugs. Make it work with any alerce feature, and fix bug causing events
        #not on alerce to give an error.
        self.alerce_features = GetAlerce(self.ztf_id)
    
    def get_features(self,train_set=False, n_components = 10, extra = pd.DataFrame()):
        Function_Parameters, Coefficients = Wavelet(self.gaussian_data)
        Coefficients = pd.concat([pd.DataFrame(Coefficients),extra],axis=1)
        pca = PCA(n_components = n_components, whiten = True)
        if train_set == True:
            Coefficients = pca.fit_transform(pd.DataFrame(Coefficients).dropna())
        if train_set == False:
            Coefficients = self.pca.transform(pd.DataFrame(Coefficients).dropna())
        labels = list()
        for i in Function_Parameters['event']:
            labels.append(Function_Parameters[Function_Parameters['event']==i]['class'].values[0])
        Function_Parameters = Function_Parameters.drop(['class'],axis=1)
        self.class_labels = pd.DataFrame(labels)
        if train_set == True:
            self.pca = pca
        self.features = pd.concat([Function_Parameters.reset_index(drop=True),pd.DataFrame(Coefficients).reset_index(drop=True)],axis=1)
    
    def fit_classifier(self):
        features = self.features.iloc[:,1:]
        labels = self.class_labels
        classifier = Classify_SN(features, labels)
    def predict(self):
        return None
    

    
    
    
    

