#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 19:08:51 2021

@author: bgarrets
"""
import pandas as pd
import numpy as np
from antares_client.search import search
from antares_client._api.models import Locus
from antares_client.search import get_by_ztf_object_id
from tqdm import tqdm
from alerce.core import Alerce

default_features = ['SPM_A', 'SPM_tau_fall', 'SPM_tau_rise', 'SPM_gamma', 'SPM_beta', 'SPM_chi','SPM_t0','LinearTrend',
                    'AndersonDarling','dmag_first_det_fid','Skew']

def GetData(ztf_id):
    try:
        return pull_from_alerce(ztf_id)
    except Exception:
        print('Event not found on ALeRCE... Trying ANTARES')
    try:
        return pull_from_antares(ztf_id)
    except Exception:
        print('Event not found on ANTARES... Skipping event')

def pull_from_antares(ztf_id):
    """
    
    Parameters
    ----------
    ztf_id : str
        ZTF ID for event.

    Returns
    -------
    Data : Pandas Data Frame
        Data Frame of lightcurve information.
    ra : float
        right ascension of event.
    dec : float
        declination of event.

    """
    
    locus = get_by_ztf_object_id(ztf_id)
    
    try:
        Data = locus.lightcurve
        ra = locus.ra
        dec = locus.dec
    except Exception:
        print("Warning: Event Could not be found")
    
    Data = Data[['ant_mag', 'ant_magerr', 'ant_mjd', 'ant_passband']].dropna().reset_index(drop=True)
    Data = Data.rename(columns = {'ant_mag':'magnitude', 'ant_magerr':'error', 'ant_mjd':'mjd',
                        'ant_passband':'band'})
    Data['band'] = Data['band'].replace(['R'], ['r'])
    return Data, ra, dec

def pull_from_alerce(ztf_id):
    """
    Parameters
    ----------
    ztf_id : str
        ZTF ID for event.

    Returns
    -------
    Data : Pandas Data Frame
        Data Frame of lightcurve information.
    ra : float
        right ascension of event.
    dec : float
        declination of event.

    """
    
    alerce = Alerce()
    
    Data = alerce.query_detections(ztf_id,
                                     format="pandas")
    ra = np.mean(Data['ra'])
    dec = np.mean(Data['dec'])
    Data = Data[['magpsf', 'sigmapsf', 'mjd', 'fid']]
    Data['fid'] = Data['fid'].replace([1,2], ['g', 'r'])
    
    Data = Data.rename(columns = {'magpsf':'magnitude', 'sigmapsf':'error','fid':'band'}).reset_index(drop=True)
        
    return Data, ra, dec
    

def alerce_features(ztf_id):
    alerce = Alerce()
    default_features = ['SPM_A', 'SPM_tau_fall', 'SPM_tau_rise', 'SPM_gamma', 'SPM_beta', 'SPM_chi','SPM_t0','LinearTrend',
                        'AndersonDarling','dmag_first_det_fid','Skew']
    features = alerce.query_features(ztf_id, format="pandas")
    
    features = features[features['name'].isin(default_features)==True]
    gband = features[features['fid']==1].reset_index(drop=True)
    rband = features[features['fid']==2].reset_index(drop=True)
    
    gband['name'] = gband['name'] + '_1'
    rband['name'] = rband['name'] + '_2'
    features = pd.concat([gband, rband]).reset_index(drop=True)
    
    Data = pd.DataFrame(list(features['value'].values)).T    
    Data.columns = list(features['name'].values)
    return Data

def GetRaDec(event):
    loop = tqdm(total = len(event), position =0, leave = False)

    alerce = Alerce()
    DF = pd.DataFrame()
    for i in event:
        loop.set_description("Fetching Object Location...".format(len(i)))
        lightcurve = alerce.query_lightcurve(i,
                                             format="json")
        ra=pd.DataFrame(lightcurve['detections'])['ra'][0]
        dec = pd.DataFrame(lightcurve['detections'])['dec'][0]
        loc = {'ra':ra, 'dec':dec}
        DF = pd.concat([DF, pd.DataFrame(loc,index=[0])])
        loop.update(1)
    return DF.reset_index(drop=True)

def GetRedshift(ra,dec,search_radius):
    client =Alerce()
    loop = tqdm(total = len(ra), position =0, leave = False)
    Redshift = pd.DataFrame()
    for i in range(len(ra)):
        try:
            loop.set_description("Finding Closest Redshift...".format(i))
            catalog_name = "GAIA/DR1"
            redshift = client.catshtm_redshift(ra[i],
                                               dec[i],
                                               search_radius,
                                               catalog_name)
            Redshift = pd.concat([Redshift, pd.DataFrame([redshift])])
            
            loop.update(1)
        except Exception:
            Redshift = pd.concat([Redshift, pd.DataFrame([-999])])
            loop.update(1)
    loop.close()
    return Redshift.reset_index(drop=True)
    
    
    
    
