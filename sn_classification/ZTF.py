# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 16:00:52 2021

@author: blgnm
"""
import light_curve
import classification
class ZTF:
    def __init__(self, ztf_id):
        lc = LightCurve(ztf_id)
        lc.Data()
        lc.gaussian_processed_regression()
        lc.get_features()
        self.lightcurve = lc
    
    def classify(self, get_probability = False):
        return Classification(self.lightcurve).classify_events(probability = get_probability)
    
    def plot_lightcurve(self, apply_gp = True):
        lc = self.lightcurve.data
        try:
            event = str(input("Enter ZTF ID:"))
            lc = lc[lc['event'] == event]
        except Exception:
            print('ZTF ID not in data base')
        
        gband = lc[lc['band']==0].reset_index(drop=True)
        rband = lc[lc['band']==1].reset_index(drop=True)
        if apply_gp == True:
            gp = self.lightcurve.gaussian_data
            gp = gp[gp['event']==event]
            gp['mjd'] = (gp['mjd'] + lc['mjd'].min())-1
            gp_gband = gp[gp['band']==4813.9]
            gp_rband = gp[gp['band']==6421.8]
            plt.errorbar(gp_gband['mjd'], gp_gband['magnitude'], yerr = gp_gband['error'], fmt = 'none', ecolor = 'green', alpha = .5)
            plt.errorbar(gp_rband['mjd'], gp_rband['magnitude'], yerr = gp_rband['error'], fmt = 'none', ecolor = 'red', alpha = .5)
            
        plt.errorbar(gband['mjd'], gband['magnitude'], yerr= gband['error'], fmt='o', ecolor= 'green', color = 'green')
        plt.errorbar(rband['mjd'], rband['magnitude'], yerr= rband['error'], fmt='o', ecolor='red', color = 'red')
        
        
        plt.gca().invert_yaxis()
        plt.xlabel('mjd')
        plt.ylabel('magnitude')
        plt.title(event)
        plt.show()
        plt.close()
    
    def plot_host_galaxy(self, source = 'pan_stars'):
        if source == 'pan_stars':
            return None
        return None
    
    
    