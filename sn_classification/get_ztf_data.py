# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:17:16 2021

@author: blgnm
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

def pull_from_antares(ztf_id):
    Supernovadf = pd.DataFrame(columns = ["mjd", "band", "magnitude", "error", "event","class"])
    loop = tqdm(total = len(ztf_id), position =0, leave = False)
    
    for i in ztf_id:
        #Pulls data for lightcurve from ANTARES
        #Skips if ANTARES doesn't have any data for that object
        locus = get_by_ztf_object_id(i)
        try:
            Data = locus.lightcurve
            ra = locus.ra
            dec = locus.dec
                
        except Exception:
            print("Warning: Event could not be found")
            pass
        #Seperates the lightcurve data into 2 seperate bands
        Data_frame1 = pd.DataFrame.from_dict(Data[Data['ant_passband']=='g']) 
        Data_frame2 = pd.DataFrame.from_dict(Data[Data['ant_passband']=='R']) 
        
        #Removes na values caused by seperating them
        Data_frame1['ant_mag'] = Data_frame1['ant_mag'].replace(np.nan, 0)
        Data_frame2['ant_mag'] = Data_frame2['ant_mag'].replace(np.nan, 0)
        Data_frame1 = Data_frame1[Data_frame1.ant_mag > 0] 
        Data_frame2 = Data_frame2[Data_frame2.ant_mag > 0] 
    
        
        MJD1 = Data_frame1['ant_mjd']
        MJD2 = Data_frame2['ant_mjd']
        MagnitudeG = Data_frame1['ant_mag'] 
        MagnitudeR = Data_frame2['ant_mag']
        
        #scales back mjd to 0 to make it easier to read
        MJD1 = MJD1 - (MJD1.min() - 1)
        MJD2 = MJD2 - (MJD2.min() - 1)
        
        
        #Defines DataFrame for G-band
        GBand = pd.DataFrame(columns = ["mjd", "band", "magnitude", "error", "event"])
        GBand["mjd"] = Data_frame1["ant_mjd"]
        GBand["band"] = pd.Series(np.zeros([len(MagnitudeG)]))
        GBand["magnitude"] = MagnitudeG    
        GBand['band'] = GBand['band'].replace(np.nan, 0)
        GBand['error'] = Data_frame1["ant_magerr"]
    
            
        #Defines DataFrame for R-band
        RBand = pd.DataFrame(columns = ["mjd", "band", "magnitude", "error", "event"])
        RBand["mjd"] = Data_frame2["ant_mjd"]
        RBand["band"] = pd.Series(np.ones([len(MagnitudeR)]))
        RBand["magnitude"] = MagnitudeR  
        RBand['band'] = RBand['band'].replace(np.nan, 1)
        RBand['error'] = Data_frame2['ant_magerr']
        num = np.zeros(len(RBand))
        num1 = np.zeros(len(GBand))
        
        #Sets event name
        GBand['event'] = num1
        RBand['event'] = num
        
        #Sets class
        GBand['class'] = num1
        RBand['class'] = num
        
        
        
        GBand['event'] = GBand['event'].replace([0], [str(i)])
        RBand['event'] = RBand['event'].replace([0], [str(i)])
        
        #Combines both Bands and joins it with final dataframe
        Both_Bands = pd.concat([GBand, RBand], axis = 0, ).reset_index(drop=True)
        Supernovadf = pd.concat([Supernovadf, Both_Bands], axis = 0).reset_index(drop=True)
    
        loop.set_description("Fetching Data...".format(len(i)))
        loop.update(1)
    
    loop.close()
    
    return Supernovadf #, pd.DataFrame({'ra':[ra],'dec':[dec]})



def GetAlerce(eventlist,feature = default_features):
    #Note: Currently Broken. Only use to get default features from alerce
    
    alerce = Alerce()
    loop = tqdm(total = len(eventlist), position =0, leave = False)
    final = pd.DataFrame()
    
    for h in eventlist:
        DF = pd.DataFrame()
        loop.set_description(f'Fetching from Alerce...'.format(len(h)))

       
        for i in feature:
            data = list()
            data2 = list()
            features = alerce.query_features(h, format="pandas")
            

            feat = features[features['name']==i]

            try:
                if list(feat['fid'].values)[0] == 12:
                    data.append(list(feat.values)[0][1])
                    data2 = list()
                if list(feat['fid'].values)[0] == 0:
                    data.append(list(feat.values)[0][1])
                    data2 = list()
                else:
                    data.append(list(feat[feat['fid']==1].values)[0][1])
                    data2.append(list(feat[feat['fid']==2].values)[0][1])
                
            except Exception:
                if list(feat['fid'].values)[0] == 12:
                    data.append(-999)
                    data2 = list()
                if list(feat['fid'].values)[0] == 0:
                    data.append(-999)
                    data2 = list()
                else:
                    data.append(-999)
                    data2.append(-999)

            loop.set_description(f'Fetching from Alerce...'.format(len(h)))
        
            DF = pd.concat([DF, pd.DataFrame(data, columns = [i])],axis=1)
            if len(data2) != 0:
                DF = pd.concat([DF, pd.DataFrame(data2, columns = [f'{i}2'])],axis=1).reset_index(drop=True)
            DF = DF.fillna(-999)
        final = pd.concat([final,DF])
        
        loop.update(1)
    loop.close()
    return final.reset_index(drop=True)

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
