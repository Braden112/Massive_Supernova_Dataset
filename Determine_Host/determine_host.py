# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:57:28 2021

@author: blgnm
"""

full_list = pd.read_csv('C:/Users/blgnm/SN_and_Host_Clasification/almost_final.csv')['event']
unlabeled = pd.DataFrame()
#%%
for i in full_list[3236:]:
    b = Flag_Object(i)[0]

# =============================================================================
#     b = b.loc[:,~b.columns.duplicated()]
#     b = b.reindex(columns = col)
# =============================================================================
    unlabeled = unlabeled.append(pd.DataFrame({'host_type':[b], 'ztf_id':[i]})).reset_index(drop=True)
    print(unlabeled)

train_list = pd.read_csv('C:/Users/blgnm/SN_and_Host_Clasification/3kList.txt')['ZTFID']
labeled = pd.DataFrame()
for i in train_list:
    b = Flag_Object(i)[0]
# =============================================================================
#     b = b.loc[:,~b.columns.duplicated()]
#     b = b.reindex(columns = col)
# =============================================================================
    labeled = labeled.append(pd.DataFrame({'host_type':[b], 'ztf_id':[i]})).reset_index(drop=True)
  
    
    print(labeled)
#%%
print(unlabeled)
#%%

#%%
for i in full_list[3236:]:
    print(i)
    break
#%%
print(unlabeled)
#%%
col = ['objID', 'primaryDetection', 'bestDetection', 'QualityFlag', 'raMean', 'decMean', 'objName', 'gPSFMag', 'gPSFMagErr', 'gApMag', 'gApMagErr', 'gKronMag', 'gKronMagErr', 'gpsfMajorFWHM', 'gpsfMinorFWHM', 'gmomentXX', 'gmomentXY', 'gmomentYY', 'gmomentR1', 'gmomentRH', 'gPSFFlux', 'gPSFFluxErr', 'gApFlux', 'gApFluxErr', 'gApRadius', 'gKronFlux', 'gKronFluxErr', 'gKronRad', 'gExtNSigma', 'rPSFMag', 'rPSFMagErr', 'rApMag', 'rApMagErr', 'rKronMag', 'rKronMagErr', 'rpsfMajorFWHM', 'rpsfMinorFWHM', 'rmomentXX', 'rmomentXY', 'rmomentYY', 'rmomentR1', 'rmomentRH', 'rPSFFlux', 'rPSFFluxErr', 'rApFlux', 'rApFluxErr', 'rApRadius', 'rKronFlux', 'rKronFluxErr', 'rKronRad', 'rExtNSigma', 'iPSFMag', 'iPSFMagErr', 'iApMag', 'iApMagErr', 'iKronMag', 'iKronMagErr', 'ipsfMajorFWHM', 'ipsfMinorFWHM', 'imomentXX', 'imomentXY', 'imomentYY', 'imomentR1', 'imomentRH', 'iPSFFlux', 'iPSFFluxErr', 'iApFlux', 'iApFluxErr', 'iApRadius', 'iKronFlux', 'iKronFluxErr', 'iKronRad', 'iExtNSigma', 'zPSFMag', 'zPSFMagErr', 'zApMag', 'zApMagErr', 'zKronMag', 'zKronMagErr', 'zpsfMajorFWHM', 'zpsfMinorFWHM', 'zmomentXX', 'zmomentXY', 'zmomentYY', 'zmomentR1', 'zmomentRH', 'zPSFFlux', 'zPSFFluxErr', 'zApFlux', 'zApFluxErr', 'zApRadius', 'zKronFlux', 'zKronFluxErr', 'zKronRad', 'zExtNSigma', 'yPSFMag', 'yPSFMagErr', 'yApMag', 'yApMagErr', 'yKronMag', 'yKronMagErr', 'ypsfMajorFWHM', 'ypsfMinorFWHM', 'ymomentXX', 'ymomentXY', 'ymomentYY', 'ymomentR1', 'ymomentRH', 'yPSFFlux', 'yPSFFluxErr', 'yApFlux', 'yApFluxErr', 'yApRadius', 'yKronFlux', 'yKronFluxErr', 'yKronRad', 'yExtNSigma', 'i-z', 'g-r', 'r-i', 'g-i', 'z-y', 'g-rErr', 'r-iErr', 'i-zErr', 'z-yErr', 'gApMag_gKronMag', 'rApMag_rKronMag', 'iApMag_iKronMag', 'zApMag_zKronMag', 'yApMag_yKronMag', '7DCD', 'label', 'distance', 'R']
print(b[list(b.columns) in col ==True])
#%%
print(Flag_Object('ZTF21aaabvit'))
#%%
print(Gaia_search(186.61304886, 31.22351592, .002))
#%%
ra = 186.61304886
dec = 31.22351592
radius = .002

coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame='icrs')
radius = u.Quantity(radius, u.deg)
j = Gaia.cone_search_async(coord, radius, table_name = 'gaiaedr3.gaia_source')
print(j)
r = j.get_results()
#%%
print(r.columns)
#%%
import pickle
import numpy as np
import pandas as pd
import scipy.interpolate as scinterp
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from panstars_query import *
from gradient_ascent import *
from host_functions import *
import warnings
warnings.filterwarnings("ignore")

import requests
import json
import csv
import pandas as pd

def Flag_Object(ztf_id):
    urlAlerce='http://api.alerce.online/ztf/v1/objects/'+str(ztf_id)+'/detections'
    responseAl = requests.get(urlAlerce)
    dataAl=responseAl.json()
    data=pd.DataFrame(dataAl)
    
    ra = float(data['ra'].mean())
    dec = float(data['dec'].mean())
    
    
    return FindHost(ra, dec)


def FindHost(ra, dec):
   
    try:
        panstars_df = query_panstars(ra, dec)
        original = panstars_df
        panstars_df = initial_filter(panstars_df)
        
    except Exception:
        print(1)
        return 'No Host', (ra,dec),pd.DataFrame([np.nan])
    
    if panstars_df.empty == True:
        #return 'No Host', None,pd.DataFrame([np.nan])
        try:
            grad,ra2,dec2 = gradient_ascent(original, ra, dec)
            print(ra2, dec2)
            
            
            host = match_host(grad, ra2[0], dec2[0])
            host['label'] =  Gaia_search(ra2[0], dec2[0], .002)
            return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host
        except Exception:
            print(2)
            return 'No Host', (ra, dec), pd.DataFrame([np.nan])
    try:
        panstars_df = star_galaxy_seperation(panstars_df)
        
    except Exception:
     
        #return 'No Host', None,pd.DataFrame([np.nan])
        try:
            grad,ra2,dec2 = gradient_ascent(original, ra, dec)
            print(ra2, dec2)
            
            
            host = match_host(grad, ra2[0], dec2[0])
            host['label'] =  Gaia_search(ra2[0], dec2[0], .002)
            return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host
        except Exception:
            print(3)
            return 'No Host', (ra,dec), pd.DataFrame([np.nan])
    if isinstance(panstars_df, str) == True:
        #return 'No Host', None,pd.DataFrame([np.nan])
        try:
            grad,ra2,dec2 = gradient_ascent(original, ra, dec)
            print(ra2, dec2)
            
            
            host = match_host(grad, ra2[0], dec2[0])
            host['label'] =  Gaia_search(ra2[0], dec2[0], .002)
            return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host
        except Exception:
            print(4)
            return 'No Host', (ra,dec), pd.DataFrame([np.nan])
    if panstars_df.empty == True:
        #return 'No Host', None, pd.DataFrame([np.nan])
        try:
            
            grad,ra2,dec2 = gradient_ascent(original, ra, dec)
            print(ra2, dec2)
            
            
            host = match_host(grad, ra2[0], dec2[0])
            host['label'] =  Gaia_search(ra2[0], dec2[0], .002)
            return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host
        except Exception:
            print(5)
            return 'No Host', (ra,dec), pd.DataFrame([np.nan])
    
    try:
        host = match_host(panstars_df, ra, dec)
    except Exception:
        #return 'No Host', None, pd.DataFrame([np.nan])
        try:
            grad,ra2,dec2 = gradient_ascent(original, ra, dec)
            print(ra2, dec2)
            
            
            host = match_host(grad, ra2[0], dec2[0])
            host['label'] =  Gaia_search(ra2[0], dec2[0], .002)
            return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host
        except Exception:
            print(6)
            return 'No Host', (ra,dec), pd.DataFrame([np.nan])
    if isinstance(host, str)==True:
        #return 'No Host', None, pd.DataFrame([np.nan])
        
        try:
            grad,ra2,dec2 = gradient_ascent(original, ra, dec)
            print(ra2, dec2)
            
            
            host = match_host(grad, ra2[0], dec2[0])
            host['label'] =  Gaia_search(ra2[0], dec2[0], .002)
            return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host
        except Exception:
            print(7)
            return 'No Host', (ra,dec), pd.DataFrame([np.nan])
    
    else:
        return host['label'].values[0], (host['raMean'].values[0], host['decMean'].values[0]), host

#%%
def initial_filter(panstars_df):
    canidates = list()
    panstars_df = panstars_df[((panstars_df['gPSFMag']!=-999) & (panstars_df['rPSFMag']!=-999) & (panstars_df['iPSFMag']!=-999))].reset_index(drop=True)
    panstars_df = panstars_df[((panstars_df['QualityFlag']!=128) & (panstars_df['primaryDetection']!=0) & (panstars_df['bestDetection']!=0))].reset_index(drop=True)

    return panstars_df.reset_index(drop=True)

def star_galaxy_seperation(filtered_panstars_df):
    model = pickle.load(open('Star_Galaxy_RealisticModel.sav','rb'))
    
    filtered_panstars_df = getColors(filtered_panstars_df)
    filtered_panstars_df = calc_7DCD(filtered_panstars_df).dropna()
    
    features = filtered_panstars_df[['7DCD','gApMag','gApMag_gKronMag','rApMag','rApMag_rKronMag','iApMag', 'iApMag_iKronMag']]
    try:
        y_pred = pd.DataFrame(model.predict_proba(features))
    except ValueError:
        return 'No Host'
    
    labels = list()
    for i in range(0,len(y_pred)):
        if y_pred[1][i] > .8:
            labels.append(1)
        else:
            labels.append(0)
    filtered_panstars_df['label'] = labels
    filtered_panstars_df['label'] = filtered_panstars_df['label'].replace([0,1],['Galaxy', 'Star'])
    
    return filtered_panstars_df.reset_index(drop=True)
    #return filtered_panstars_df[filtered_panstars_df['label']==0].drop(['label'],axis=1).reset_index(drop=True)

def match_host(seperated_panstars_df, ra, dec):
    dlr = pd.DataFrame()
    seperated_panstars_df = seperated_panstars_df.drop_duplicates(subset = ['objID'], keep= 'first', ignore_index = True).reset_index(drop=True)
    for i in seperated_panstars_df['objID']:
        canidate = seperated_panstars_df[seperated_panstars_df['objID']==i].reset_index(drop=True)
        SN_ra = Angle(ra, unit = u.deg)
        SN_dec = Angle(dec, unit = u.deg)
        
        best_filter = choose_band_SNR(canidate)
        
        r_a = canidate[best_filter + 'KronRad']
        
        
        dist, R = calc_DLR(SN_ra, SN_dec, canidate['raMean'], canidate['decMean'], float(r_a), canidate, best_filter)
        
        dlr = dlr.append(pd.DataFrame({'distance':[float(dist)], 'R':[float(R)]})).reset_index(drop=True)
    dlr = pd.concat([seperated_panstars_df, dlr],axis=1)
    try:
        dlr_gal = dlr[dlr['label']=='Galaxy']
        dlr_star = dlr[dlr['label']=='Star']
        
        dlr_gal = dlr_gal[dlr_gal['R'] < 5].reset_index(drop=True)
        dlr_star = dlr_star[dlr_star['R'] < 1].reset_index(drop=True)
        dlr = pd.concat([dlr_gal, dlr_star])
    except Exception:
        dlr = dlr[dlr['R'] < 5].reset_index(drop=True)
    if dlr.empty == True:
        return 'No Host'
    
    host_galaxy = dlr[dlr['R']==np.min(dlr['R'])]
    return host_galaxy
