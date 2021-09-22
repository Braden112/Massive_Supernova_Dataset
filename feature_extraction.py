#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:13:20 2021

@author: bgarrets
"""

import pandas as pd
import numpy as np
import pywt
from scipy import integrate
from tqdm import tqdm

def Wavelet(Gaussian_Data, WaveletType = 'sym2', Date = None, length = 15000):
    """
    

    Parameters
    ----------
    
    Data : Pandas Data Frame, optional
        Pandas DataFrame processed by Data(). The default is None.
    WaveletType : Str, optional
        Type of Wavelet transformation to be used. The default is 'sym2'.
    classification : Boolean, optional
        If you are making a training set to True (I always keep it True personally, not sure if it works otherwise). 
        The default is True.
    Date : Integer, optional
        How many days you want the classifier to look at. The default is None. The default is None.
    length : Integer, optional
        Set maximum event length; all events longer than set length are filtered out. The default is 150.

    Returns
    -------
    Function_Parameters : Pandas DataFrame
        Event Information such as ZTF ID and classification.
    Coefficients : Numpy Array
        List of Wavelet transformation Coefficients.

    """
   
    Function_Parameters = pd.DataFrame()
    Coefficients = list()
    
    
    Data_uqlst = Gaussian_Data['event'].unique()
    loop = tqdm(total = len(Gaussian_Data['event'].unique()), position =0, leave = False)
    
    #loops through each event
    for i in Data_uqlst:
        
        
        one_event_gaussian = Gaussian_Data[Gaussian_Data['event']==i]
        

        x = one_event_gaussian['mjd'].astype(float)
        y = one_event_gaussian['magnitude'].astype(float)                    
        y_err = one_event_gaussian['error'].astype(float)                   
        signal = y.values.squeeze()
        
        
        Area = integrate.simps(y, x)
        
        ca = np.array(pywt.swt(np.array(signal), WaveletType, level = 2, axis = 0))

        npoints=len(ca[0, 0, :])
        coefficients =ca.reshape(2*2, npoints)

        
        
        Features = pd.DataFrame(data = {'event': str(i), 
                                        'delta':y.values.max()-y.values.min(), 'variance':y.var(), 
                                        'duration': max(Gaussian_Data[Gaussian_Data['event']==i]['mjd'])-min(Gaussian_Data[Gaussian_Data['event']==i]['mjd']),
                                        'area':Area}, index=[0])
       

        Coefficients.append(coefficients.flatten())

        Function_Parameters = pd.concat([Function_Parameters, Features], axis =0 )
        Function_Parameters = Function_Parameters.reset_index(drop=True)
        loop.set_description("Computing Wavelet Transform...".format(len(i)))
        loop.update(1)
    loop.close()
    
    return Function_Parameters, Coefficients
