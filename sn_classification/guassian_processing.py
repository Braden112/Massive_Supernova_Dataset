# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 12:52:44 2021

@author: blgnm
"""
import george
from tqdm import tqdm
import pandas as pd
import numpy as np
from astropy.stats import biweight_location
from scipy.optimize import minimize


def Multi_Band_GP(x_range, x, y, y_err, dim, n_samples = False, sampling = False):
    """ Considers cross corrolations between multiple bands as dims, prone to holding the order of the bands too rigidly """
    """ Will optimize for 'best' parameters when given no parameters """
    """ x = mjd, y and y_err = measurment, dim and dim_err = wavelength in nm """
    length_scale = 20
    signal_to_noises = (np.abs(y) / np.sqrt(np.power(y_err,2) + (1e-2 * np.max(y))**2))
    scale = np.abs(y[signal_to_noises.argmax()])
    kernel = ((0.5 * scale)**2 * george.kernels.Matern32Kernel([length_scale**2, 6000**2], ndim=2))
    kernel.freeze_parameter('k2:metric:log_M_1_1')
    kernel.freeze_parameter('k1:log_constant') #Fixed Scale
    x_data = np.vstack([x, dim]).T
    gp = george.GP(kernel, mean = biweight_location(y))
    guess_parameters = gp.get_parameter_vector()
    gp.compute(x_data, y_err)
    x_pred = np.linspace(x.min(), x.max(), n_samples)
    x_pred = np.vstack([x, dim]).T
    pred, pred_var = gp.predict(y, x_pred, return_var=True)
    # bounds = [(0, np.log(1000**2))]
    def neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.log_likelihood(y)
    def grad_neg_ln_like(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y)
    result = minimize(
            neg_ln_like,
            gp.get_parameter_vector(),
            jac=grad_neg_ln_like,
            # bounds=bounds
            )
    if result.success:
            gp.set_parameter_vector(result.x)
    else:
            gp.set_parameter_vector(guess_parameters)    
    gp.set_parameter_vector(result.x)
    # print(kernel.get_parameter_vector(True))
    #print("\nFinal ln-likelihood: {0:.2f}".format(gp.log_likelihood(y)))
    if n_samples != False:
        x_pred = np.vstack([np.array(list(np.linspace(x_range.min(), x_range.max(), n_samples))*np.unique(dim).size),
                            np.array(np.sort(list(np.unique(dim))*n_samples))]).T 
        # x_pred = np.vstack([np.array(list(np.linspace(x_range.min(), x_range.max(), n_samples))*6),
        #                     np.array(np.sort([357, 476, 621, 754, 870, 1004]*n_samples))]).T 
        pred, pred_var = gp.predict(y, x_pred, return_var=True)
        output = [x_pred[:,0], pred, np.sqrt(pred_var), x_pred[:,1], []]
        return output
    elif sampling != False:
        x_pred = np.vstack([np.array(sampling[0]),
                            np.array(sampling[1])]).T  
        pred, pred_var = gp.predict(y, x_pred, return_var=True)
        output = [x_pred[:,0], pred, np.sqrt(pred_var), x_pred[:,1], []]
        return output

def band_to_color(inp):
    labels = [4813.9, 6421.8, 621, 754, 870, 1004]
    # labels = [0,1,2,3,4,5]
    labels_2=['green', 'red', 'goldenrod', 'blue', 'pink', 'grey']
    outlst = []
    for x in inp:
        out = labels.index(int(x))
        out = labels_2[out]
        outlst.append(out)
    return outlst

def band_to_wvlg(inp):
    labels = [0,1,2,3,4,5]
    labels_2=[4813.9, 6421.8, 621.5, 754.5, 870.75, 1004.0]
    outlst = []
    for x in inp:
        out = labels.index(int(x))
        out = labels_2[out]
        outlst.append(out)
    return outlst

def expfun(x, a, b):
    return np.multiply(np.exp(np.multiply(x, b)), a)

def randomoff(inp, off = 0.25):
    outlist = []
    for i in inp:
        value = random.random()
        outlist += [i+value*off]
    return outlist

def Spectra_Model():
    return 0

def gaussian_regression(light_curve_data, maximum_sn_length = None, n_samples = 100 ,classification = True):
    #if data is None:
     #   if self.data is None:
      #      sf = self.Data()
       # else:
            #sf = self.data
    #else:
     #   sf = data
    sf = light_curve_data
    gaussian_light_curve = pd.DataFrame()
    pd.options.mode.chained_assignment = None  # default='warn'
    SN_uqlst = sf.event.unique()
    loop = tqdm(total = len(SN_uqlst), position =0, leave = False)
    
    #Loops through each event in the dataset and performs gaussian regression on them
    for i in SN_uqlst:
        SNdf = sf[sf['event']==i]
        #scales back date range so that the first date is 1 (makes it easier to read)
        SNdf['mjd'] = SNdf['mjd'] - (SNdf['mjd'].min() -1)
        #Filters out any data points past the Date Range parameter
        if maximum_sn_length is not None:
            SNdf = SNdf[SNdf['mjd'] < maximum_sn_length]
        b = SNdf['band'].unique() == np.array([0.0, 1.0])
        #Skips event if there isn't data in both bands
        if b[0] != True or b[1] != True:
            continue
        
        mjdrange = np.asarray([min(SNdf['mjd'].tolist()),max(SNdf['mjd'].tolist())])
        #Calculates Gaussian Regression for event
        D = Multi_Band_GP(x_range = mjdrange, x=SNdf['mjd'].to_numpy(),
                          y=SNdf['magnitude'].to_numpy(), y_err=SNdf['error'].to_numpy(),
                          dim=band_to_wvlg(SNdf['band'].to_numpy()),
                          n_samples= n_samples)
        GaussianFitted = pd.DataFrame()
        GaussianFitted['mjd'] = D[0]
        GaussianFitted['magnitude'] = D[1]
        GaussianFitted['error'] = D[2]
        GaussianFitted['band'] = D[3]
        y = pd.Series(data=np.zeros(1000)).astype(int)
        y = y.replace(0,i)
        GaussianFitted['event'] = y
        if classification == True:
            x = pd.Series(data = np.zeros(1000)).astype(int)
            x = x.replace(0, SNdf['class'].unique()[0])
            GaussianFitted['class'] = x
        gaussian_light_curve = pd.concat([gaussian_light_curve, GaussianFitted])
        loop.set_description("Computing GPR...".format(len(i)))
        loop.update(1)
    loop.close()
    return gaussian_light_curve
