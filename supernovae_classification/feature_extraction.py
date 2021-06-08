# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:19:34 2021

@author: blgnm
"""
import pandas as pd
import numpy as np
import pywt

def Wavelet(Gaussian_Data, WaveletType = 'sym2', classification = True, Date = None, length = 150):
    """
    

    Parameters
    ----------
    Note: This version Processes both bands together, see Wavelet() for seperate band processing
    
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
   
    from tqdm import tqdm
    Function_Parameters = pd.DataFrame()
    Coefficients = list()
    
    #If data isn't provided, it pulls data from ANTARES and performs guassian regression on it 
    #if Data is None:
     #   if self.data is None:
      #      Data = self.Data()
       # else:
        #    Data = self.data
    #if self.GaussianData is None:
     #   Gaussian = self.GaussianRegression(data = Data, DateRange = Date)
    #else:
     #   Gaussian = self.GaussianData
    
    Data_uqlst = Gaussian_Data['event'].unique()
    loop = tqdm(total = len(Gaussian_Data['event'].unique()), position =0, leave = False)
    
    #loops through each event
    for i in Data_uqlst:
        #Skips event if there isn't data in both bands
        #b = Data[(Data['event']==i)]['band'].unique() == np.array([0.0, 1.0])
        #if b[0] != True or b[1] != True:
         #   continue
        #Skips data if outside range of length
        if max(Gaussian_Data[Gaussian_Data['event']==i]['mjd'])-min(Gaussian_Data[Gaussian_Data['event']==i]['mjd']) > length:
            #print(len(Data[Data['event']==i]['mjd']))
            continue
        
        one_event_gaussian = Gaussian_Data[Gaussian_Data['event']==i]
        
        
            
        if classification == True:
            Class = one_event_gaussian['class']

        x = one_event_gaussian['mjd'].astype(float)
        y = one_event_gaussian['magnitude'].astype(float)                    
        y_err = one_event_gaussian['error'].astype(float)                   
        signal = y.values.squeeze()
        if len(signal) == 0:
            continue
        from scipy import integrate
        Area = integrate.simpson(y, x)
        
        ca = np.array(pywt.swt(np.array(signal), WaveletType, level = 2, axis = 0))

        npoints=len(ca[0, 0, :])
        coefficients =ca.reshape(2*2, npoints)
        
        
        Features = pd.DataFrame(data = {'event': str(i), 
                                        'delta':y.values.max()-y.values.min(), 'variance':y.var(), 
                                        'duration': max(Gaussian_Data[Gaussian_Data['event']==i]['mjd'])-min(Gaussian_Data[Gaussian_Data['event']==i]['mjd']),
                                        'area':Area}, index=[0])
        if classification == True:
            Features['class'] = Class.unique()[0]

        Coefficients.append(coefficients.flatten())

        Function_Parameters = pd.concat([Function_Parameters, Features], axis =0 )
        Function_Parameters = Function_Parameters.reset_index(drop=True)
        loop.set_description("Computing Wavelet Transform...".format(len(i)))
        loop.update(1)
    loop.close()
    Function_Parameters['class'] = Function_Parameters['class'].replace(['SN Ia', 'SN II', 'SN Ib/c', 'SLSN'], [0,1,2,3])
    #Function_Parameters['class'] = Function_Parameters['class'].replace(['SN Ia', 'SN II', 'SN IIn', 'SN IIP', 'SN Ia-91T', 'SLSN-I', 'SLSN-II', 'SN Ic', 'SN Ib', 'SN Ic-BL', 'SN IIb', 'SN Ia-pec', 'SN Ibn', 'SN Ia-91bg'], [0,1,2,3,4,5,6,7,8,9, 10,11,12,13])
    
    return Function_Parameters, Coefficients

def DimensionalityReduction2(Coefficients =None, labels=None, Extra = None, smot = False, n = 20, Trainset = True):
    """
    

    Parameters
    ----------
    Use this version if you used Wavelet2() (Multiband processing)
    
    Coefficients : Pandas Data Frame, optional
        Provide your own wavelet coefficients. The default is None.
    labels : Pandas Data Frame, optional
        Provide your own labels. The default is None.
    smot : Boolean, optional
        Choose whether or not to use SMOTE. The default is False.
    n : Integer, optional
        Output Dimension. The default is 20.
    Trainset : Boolean, optional
        Specify if this is the training set or if its unlabeled data. The default is True.

    Returns
    -------
    Pandas Data Frame
        Pandas Data Frame of PCA reduced Wavelet Coefficients.
    Function
        If Trainset = True, returns PCA() to use on unlabeled data.


    """
    if Coefficients is None:
        if self.wavelet_transform is None:
            labels, Coefficients = self.Wavelet2()
        else:
            Coefficients = self.wavelet_transform
            labels = self.metadata
    
    else:    
        labels = labels
        Coefficients = Coefficients
        Extra = Extra

        
    
    Coefficients = pd.concat([pd.DataFrame(data=labels),pd.DataFrame(data=Coefficients)],axis=1)

    Coeff = Coefficients.iloc[:,6:]
    
    pca = PCA(n_components = n, whiten = True)
    if smot == True:
        sm = SMOTE()
        Coeff, labels= sm.fit_resample(Coeff, Coefficients['class'].ravel())
    
    if Trainset == True:
        final = pca.fit_transform((Coeff))
    #RBand2, GBand2 = pd.DataFrame(data = {'Rdelta': RBand['delta'], 'Rvariance': RBand['variance']}), pd.DataFrame(data = {'Gdelta':GBand['delta'], 'Gvariance': GBand['variance']})
    if smot == True:
        events =pd.DataFrame(data = {'class': labels}).reset_index(drop=True)
    if smot == False:
        events =pd.DataFrame(data = {'event': Coefficients['event'], 'class': Coefficients['class']}).reset_index(drop=True)
    if Trainset == True:
        ProcessedData = pd.concat([events, pd.DataFrame(final)],axis=1)
        self.ProcessedData = ProcessedData
        self.pca = pca
        return ProcessedData, pca
    if Trainset == False:
        return pd.concat([events, pd.DataFrame(data = Coeff).reset_index(drop=True)],axis=1)