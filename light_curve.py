#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:07:57 2021

@author: bgarrets
"""

from guassian_processing import *
from get_ztf_data import *
from feature_extraction import *
from sklearn.decomposition import PCA
import sfdmap
from pickle import load
from get_ztf_data import pull_from_alerce, alerce_features

class LightCurve:
    def __init__(self,ztf_id = None,light_curve_data=None):
        self.ztf_id = ztf_id
        self.data = light_curve_data
        self.gaussian_data = None
        self.alerce_features = None
        self.features = None
        self.pca = load(open('sn_pca.sav', 'rb'))
        self.class_labels = None
        self.ra = None
        self.dec = None
        self.ebv = None
        
    def Data(self):
        event = self.ztf_id
        data= pd.DataFrame()
        ztf = list()
        for i in event:
            x = pull_from_alerce(i)[0]
            x['band'] = x['band'].replace(['g','r'],[0,1])
            for g in range(len(x['magnitude'].values)):
                ztf.append(i)
            data = data.append(x).reset_index(drop=True)
        
        data['event'] = ztf
        self.data = data
            
    def get_location(self):
        location = GetRaDec(self.ztf_id).reset_index(drop=True)
        self.ra = location['ra']
        self.dec = location['dec']
    def milky_way_extinction(self):
        dustmap = sfdmap.SFDMap("sfddata-master/")
        ebv = list()
        for i in range(0, len(self.ra)):
            ebv.append(dustmap.ebv(self.ra[i],self.dec[i]))
        self.ebv = ebv
    def gaussian_processed_regression(self,maximum_sn_length = 200, n_samples = 100 ,classification = True):
        self.gaussian_data = gaussian_regression(self.data)
    
    def get_alerce_features(self):
        #Note to self: Go back and fix bugs. Make it work with any alerce feature, and fix bug causing events
        #not on alerce to give an error.
        self.alerce_features, self.missed = GetAlerce(self.ztf_id)
    
    def get_features(self,train_set=False, n_components = 10, extra = pd.DataFrame()):
        Function_Parameters, Coefficients = Wavelet(self.gaussian_data)
        
        Function_Parameters = Function_Parameters.loc[pd.DataFrame(Coefficients).dropna().index].dropna()
        Coefficients = pd.DataFrame(Coefficients).dropna()
        
        pca = PCA(n_components = n_components, whiten = True)
        if train_set == True:
            Coefficients = pca.fit_transform(pd.DataFrame(Coefficients).dropna())
        if train_set == False:
            Coefficients = self.pca.transform(pd.DataFrame(Coefficients).dropna())
        labels = list()
       
        self.class_labels = pd.DataFrame(labels)
        
        if train_set == True:
            self.pca = pca
        self.features = pd.concat([Function_Parameters.reset_index(drop=True),pd.DataFrame(Coefficients).reset_index(drop=True)],axis=1)
    
    def predict(self):
        return None
    
    def extinction_correction(self):
        data = self.data
        new_data= pd.DataFrame()
        for i in range(0,len(data['event'].unique())):
            event = data['event'].unique()[i]
            event = data[data['event']==event]
            A_g = self.ebv[i]*3.303
            A_r = self.ebv[i]*2.088
            g_corrected = A_g*1.21
            r_corrected = A_r*.848
            
            gband = event[event['band']==0] 
            rband = event[event['band']==1]
            gband['magnitude'] = gband['magnitude'] - g_corrected
            rband['magnitude'] = rband['magnitude'] - r_corrected
            new_data = new_data.append([gband, rband]).reset_index(drop=True)
        self.data =  new_data
    

    

