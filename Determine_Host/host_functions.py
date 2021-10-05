# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:57:47 2021

@author: blgnm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 00:19:13 2021

@author: bgarrets
"""

from matplotlib import colors
from scipy import ndimage
from astropy.wcs import WCS
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
from astropy.io import ascii
from astropy.table import Table
import requests
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
from collections import OrderedDict
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.io import ascii
from astropy.table import Table
from panstars_query import *
import sys
import re
import numpy as np
import pylab
import json
import requests
import scipy.interpolate as scinterp
import pickle
try: # Python 3.x
    from urllib.parse import quote as urlencode
    from urllib.request import urlretrieve
except ImportError:  # Python 2.x
    from urllib import pathname2url as urlencode
    from urllib import urlretrieve

try: # Python 3.x
    import http.client as httplib 
except ImportError:  # Python 2.x
    import httplib   


def calc_DLR(ra_SN, dec_SN, ra_host, dec_host, r_a, source, best_band):
    # EVERYTHING IS IN ARCSECONDS

    ## taken from "Understanding Type Ia Supernovae Through Their Host Galaxies..." by Gupta
    #https://repository.upenn.edu/cgi/viewcontent.cgi?referer=https://www.google.com/&httpsredir=1&article=1916&context=edissertations
    xr = np.abs(ra_SN.deg - float(ra_host))*3600
    yr = np.abs(dec_SN.deg - float(dec_host))*3600

    SNcoord = SkyCoord(ra_SN, dec_SN, frame='icrs')
    hostCoord = SkyCoord(ra_host*u.deg, dec_host*u.deg, frame='icrs')
    sep = SNcoord.separation(hostCoord)
    dist = sep.arcsecond
    
    badR = 10000000000 # if we don't have spatial information, get rid of it #this
    # is good in that it gets rid of lots of artifacts without radius information
    #dist = float(np.sqrt(xr**2 + yr**2))


    XX = best_band + 'momentXX'
    YY = best_band + 'momentYY'
    XY = best_band + 'momentXY'

    if (float(source[XX]) != float(source[XX])) | (float(source[XY]) != float(source[XY])) | \
        (float(source[YY]) != float(source[YY])):
        return dist, badR

    U = float(source[XY])
    Q = float(source[XX]) - float(source[YY])
    if Q == 0:
        return dist, badR

    phi = 0.5*np.arctan(U/Q)
    kappa = Q**2 + U**2
    a_over_b = (1 + kappa + 2*np.sqrt(kappa))/(1 - kappa)

    gam = np.arctan(yr/xr)
    theta = phi - gam

    DLR = r_a/np.sqrt(((a_over_b)*np.sin(theta))**2 + (np.cos(theta))**2)

    R = float(dist/DLR)

    if (R != R):
        return dist, badR

    return dist, R

def choose_band_SNR(host_df):
    bands = ['g', 'r', 'i', 'z', 'y']
    try:
        gSNR = float(1/host_df["gPSFMagErr"])
        rSNR = float(1/host_df["rPSFMagErr"])
        iSNR = float(1/host_df["iPSFMagErr"])
        zSNR = float(1/host_df["zPSFMagErr"])
        ySNR = float(1/host_df["yPSFMagErr"])

        SNR = np.array([gSNR, rSNR, iSNR, zSNR, ySNR])
        i = np.nanargmax(SNR)
    except:
        #if we have issues getting the band with the highest SNR, just use 'r'-band
        i = 1
    return bands[i]

def getColors(df):
    df.replace(-999, np.nan, inplace=True)
    df.replace(999, np.nan, inplace=True)
    # create color attributes for all hosts
    df["i-z"] = df["iApMag"] - df["zApMag"]
    df["g-r"]= df["gApMag"] - df["rApMag"]
    df["r-i"]= df["rApMag"] - df["iApMag"]
    df["g-i"] = df["gApMag"] - df["iApMag"]
    df["z-y"] = df["zApMag"] - df["yApMag"]

    df['g-rErr'] = np.sqrt(df['gApMagErr']**2 + df['rApMagErr']**2)
    df['r-iErr'] = np.sqrt(df['rApMagErr']**2 + df['iApMagErr']**2)
    df['i-zErr'] = np.sqrt(df['iApMagErr']**2 + df['zApMagErr']**2)
    df['z-yErr'] = np.sqrt(df['zApMagErr']**2 + df['yApMagErr']**2)

    # To be sure we're getting physical colors
    df.loc[df['i-z'] > 100, 'i-z'] = np.nan
    df.loc[df['i-z'] < -100, 'i-z'] = np.nan
    df.loc[df['g-r'] > 100, 'i-z'] = np.nan
    df.loc[df['g-r'] < -100, 'i-z'] = np.nan

    # and PSF - Kron mag "morphology" information
    df["gApMag_gKronMag"] = df["gApMag"] - df["gKronMag"]
    df["rApMag_rKronMag"] = df["rApMag"] - df["rKronMag"]
    df["iApMag_iKronMag"] = df["iApMag"] - df["iKronMag"]
    df["zApMag_zKronMag"] = df["zApMag"] - df["zKronMag"]
    df["yApMag_yKronMag"] = df["yApMag"] - df["yKronMag"]
    #df["gApMag_rApMag"] = df["gApMag"] - df["rApMag"]
    #df["iApMag_zApMag"] = df["iApMag"] - df["zApMag"]

    # to be sure we're getting physical mags
    df.loc[df['iApMag_iKronMag'] > 100, 'iApMag_iKronMag'] = np.nan
    df.loc[df['iApMag_iKronMag'] < -100, 'iApMag_iKronMag'] = np.nan
    df.loc[df['iApMag'] > 100, 'iApMag'] = np.nan
    df.loc[df['iApMag'] < -100, 'iApMag'] = np.nan
    return df

def calc_7DCD(df):
    #read the stellar locus table from SDSS
    df.replace(999.00, np.nan)
    df.replace(-999.00, np.nan)

    #stream = pkg_resources.resource_stream(__name__, 'tonry_ps1_locus.txt')
    
    #skt = at.Table.read(stream, format='ascii')

    skt = Table.from_pandas(pd.read_csv('torny_ps1_locus.csv'))
    

    gr = scinterp.interp1d(skt['ri'], skt['gr'], kind='cubic', fill_value='extrapolate')
    iz = scinterp.interp1d(skt['ri'], skt['iz'], kind='cubic', fill_value='extrapolate')
    zy = scinterp.interp1d(skt['ri'], skt['zy'], kind='cubic', fill_value='extrapolate')
    ri = np.arange(-0.4, 2.01, 0.001)

    gr_new = gr(ri)
    iz_new = iz(ri)
    zy_new = zy(ri)

    bands = ['g', 'r', 'i', 'z']

    #adding the errors in quadrature
    df["g-rErr"] =  np.sqrt(df["gApMagErr"].astype('float')**2 + df["rApMagErr"].astype('float')**2)
    df["r-iErr"] =  np.sqrt(df["rApMagErr"].astype('float')**2 + df["iApMagErr"].astype('float')**2)
    df["i-zErr"] =  np.sqrt(df["iApMagErr"].astype('float')**2 + df["zApMagErr"].astype('float')**2)
    df['z-yErr'] =  np.sqrt(df['zApMagErr'].astype('float')**2 + df['yApMagErr'].astype('float')**2)

    df["7DCD"] = np.nan
    df.reset_index(drop=True, inplace=True)
    for i in np.arange(len(df["i-z"])):
        temp_7DCD = []

        temp_7DCD_1val_gr = (df.loc[i,"g-r"] - gr_new)**2/df.loc[i, "g-rErr"]
        temp_7DCD_1val_ri = (df.loc[i,"r-i"] - ri)**2 /df.loc[i, "r-iErr"]
        temp_7DCD_1val_iz = (df.loc[i,"i-z"] - iz_new)**2/df.loc[i, "i-zErr"]
        temp_7DCD_1val_zy = (df.loc[i,"z-y"] - zy_new)**2/df.loc[i, "z-yErr"]

        temp_7DCD_1val = temp_7DCD_1val_gr + temp_7DCD_1val_ri + temp_7DCD_1val_iz + temp_7DCD_1val_zy

        df.loc[i,"7DCD"] = np.nanmin(np.array(temp_7DCD_1val))
    return df

def ps1cone(ra,dec,radius,table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a cone search of the PS1 catalog
    
    Parameters
    ----------
    ra (float): (degrees) J2000 Right Ascension
    dec (float): (degrees) J2000 Declination
    radius (float): (degrees) Search radius (<= 0.5 degrees)
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2)
    """
    
    data = kw.copy()
    data['ra'] = ra
    data['dec'] = dec
    data['radius'] = radius
    return ps1search(table=table,release=release,format=format,columns=columns,
                    baseurl=baseurl, verbose=verbose, **data)


def ps1search(table="mean",release="dr1",format="csv",columns=None,
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False,
           **kw):
    """Do a general search of the PS1 catalog (possibly without ra/dec/radius)
    
    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    format: csv, votable, json
    columns: list of column names to include (None means use defaults)
    baseurl: base URL for the request
    verbose: print info about request
    **kw: other parameters (e.g., 'nDetections.min':2).  Note this is required!
    """
    
    data = kw.copy()
    if not data:
        raise ValueError("You must specify some parameters for search")
    checklegal(table,release)
    if format not in ("csv","votable","json"):
        raise ValueError("Bad value for format")
    url = "{baseurl}/{release}/{table}.{format}".format(**locals())
    if columns:
        # check that column values are legal
        # create a dictionary to speed this up
        dcols = {}
        for col in ps1metadata(table,release)['name']:
            dcols[col.lower()] = 1
        badcols = []
        for col in columns:
            if col.lower().strip() not in dcols:
                badcols.append(col)
        if badcols:
            raise ValueError('Some columns not found in table: {}'.format(', '.join(badcols)))
        # two different ways to specify a list of column values in the API
        # data['columns'] = columns
        data['columns'] = '[{}]'.format(','.join(columns))

# either get or post works
#    r = requests.post(url, data=data)
    r = requests.get(url, params=data)

    if verbose:
        print(r.url)
    r.raise_for_status()
    if format == "json":
        return r.json()
    else:
        return r.text


def checklegal(table,release):
    """Checks if this combination of table and release is acceptable
    
    Raises a VelueError exception if there is problem
    """
    
    releaselist = ("dr1", "dr2")
    if release not in ("dr1","dr2"):
        raise ValueError("Bad value for release (must be one of {})".format(', '.join(releaselist)))
    if release=="dr1":
        tablelist = ("mean", "stack")
    else:
        tablelist = ("mean", "stack", "detection")
    if table not in tablelist:
        raise ValueError("Bad value for table (for {} must be one of {})".format(release, ", ".join(tablelist)))


def ps1metadata(table="mean",release="dr1",
           baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"):
    """Return metadata for the specified catalog and table
    
    Parameters
    ----------
    table (string): mean, stack, or detection
    release (string): dr1 or dr2
    baseurl: base URL for the request
    
    Returns an astropy table with columns name, type, description
    """
    
    checklegal(table,release)
    url = "{baseurl}/{release}/{table}/metadata".format(**locals())
    r = requests.get(url)
    r.raise_for_status()
    v = r.json()
    # convert to astropy table
    tab = Table(rows=[(x['name'],x['type'],x['description']) for x in v],
               names=('name','type','description'))
    return tab


def mastQuery(request):
    """Perform a MAST query.

    Parameters
    ----------
    request (dictionary): The MAST request json object

    Returns head,content where head is the response HTTP headers, and content is the returned data
    """
    
    server='mast.stsci.edu'

    # Grab Python Version 
    version = ".".join(map(str, sys.version_info[:3]))

    # Create Http Header Variables
    headers = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/plain",
               "User-agent":"python-requests/"+version}

    # Encoding the request as a json string
    requestString = json.dumps(request)
    requestString = urlencode(requestString)
    
    # opening the https connection
    conn = httplib.HTTPSConnection(server)

    # Making the query
    conn.request("POST", "/api/v0/invoke", "request="+requestString, headers)

    # Getting the response
    resp = conn.getresponse()
    head = resp.getheaders()
    content = resp.read().decode('utf-8')

    # Close the https connection
    conn.close()

    return head,content


def resolve(name):
    """Get the RA and Dec for an object using the MAST name resolver
    
    Parameters
    ----------
    name (str): Name of object

    Returns RA, Dec tuple with position"""

    resolverRequest = {'service':'Mast.Name.Lookup',
                       'params':{'input':name,
                                 'format':'json'
                                },
                      }
    headers,resolvedObjectString = mastQuery(resolverRequest)
    resolvedObject = json.loads(resolvedObjectString)
    # The resolver returns a variety of information about the resolved object, 
    # however for our purposes all we need are the RA and Dec
    try:
        objRa = resolvedObject['resolvedCoordinate'][0]['ra']
        objDec = resolvedObject['resolvedCoordinate'][0]['decl']
    except IndexError as e:
        raise ValueError("Unknown object '{}'".format(name))
    return (objRa, objDec)