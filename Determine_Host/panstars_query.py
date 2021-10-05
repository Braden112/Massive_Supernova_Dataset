# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 15:58:26 2021

@author: blgnm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 00:16:54 2021

@author: bgarrets
"""

import requests
from astropy.io import ascii
from astropy.table import Table
from astropy.io import fits
import matplotlib
from astropy import units as u
from astropy.coordinates import SkyCoord
#from astroquery.ned import Ned
import re
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def getimages(ra,dec,size=240,filters="grizy", type='stack'):

    """Query ps1filenames.py service to get a list of images
    ra, dec = position in degrees
    size = image size in pixels (0.25 arcsec/pixel)
    filters = string with filters to include
    Returns a table with the results
    """

    service = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
    url = ("{service}?ra={ra}&dec={dec}&size={size}&format=fits"
           "&filters={filters}&type={type}").format(**locals())
    table = Table.read(url, format='ascii')
    return table


def query_panstars(ra, dec, radius = 30):
    radius = radius/3600.0
        
        
# =============================================================================
#     columns = """objID,primaryDetection,gPSFMag,rPSFMag,iPSFMag,bestDetection,QualityFlag,
#                  iApMag,gApMag,rApMag,zApMag,yApMag,gApMagErr,rApMagErr,iApMagErr,zApMagErr,
#                  yApMagErr,gKronMag,rKronMag,iKronMag,zKronMag,yKronMag,raMean,decMean,
#                  gmomentXX,gmomentXY,gmomentYY,rmomentXX,rmomentXY,rmomentYY,imomentXX,
#                  imomentXY,imomentYY,zmomentXX,zmomentXY,zmomentYY,ymomentXX,ymomentXY,
#                  ymomentYY,gKronRad,rKronRad,iKronRad,zKronRad,yKronRad, objName""".split(',')
# =============================================================================
    columns = ['objID','primaryDetection','bestDetection','QualityFlag','raMean','decMean',
               'objName','gPSFMag', 'gPSFMagErr', 'gApMag', 'gApMagErr', 'gKronMag', 'gKronMagErr', 'gpsfMajorFWHM', 
           'gpsfMinorFWHM', 'gmomentXX', 'gmomentXY', 'gmomentYY', 'gmomentR1', 'gmomentRH', 'gPSFFlux', 
           'gPSFFluxErr', 'gApFlux', 'gApFluxErr', 'gApRadius', 'gKronFlux', 'gKronFluxErr', 'gKronRad', 
           'gExtNSigma', 'rPSFMag', 'rPSFMagErr', 'rApMag', 'rApMagErr', 'rKronMag', 'rKronMagErr', 'rpsfMajorFWHM', 
           'rpsfMinorFWHM', 'rmomentXX', 'rmomentXY', 'rmomentYY', 'rmomentR1', 'rmomentRH', 'rPSFFlux', 
           'rPSFFluxErr', 'rApFlux', 'rApFluxErr', 'rApRadius', 'rKronFlux', 'rKronFluxErr', 'rKronRad', 
           'rExtNSigma', 'iPSFMag', 'iPSFMagErr', 'iApMag', 'iApMagErr', 'iKronMag', 'iKronMagErr', 'ipsfMajorFWHM', 
           'ipsfMinorFWHM', 'imomentXX', 'imomentXY', 'imomentYY', 'imomentR1', 'imomentRH', 'iPSFFlux', 'iPSFFluxErr', 
           'iApFlux', 'iApFluxErr', 'iApRadius', 'iKronFlux', 'iKronFluxErr', 'iKronRad', 'iExtNSigma', 'zPSFMag', 
           'zPSFMagErr', 'zApMag', 'zApMagErr', 'zKronMag', 'zKronMagErr', 'zpsfMajorFWHM', 'zpsfMinorFWHM', 
           'zmomentXX', 'zmomentXY', 'zmomentYY', 'zmomentR1', 'zmomentRH', 'zPSFFlux', 'zPSFFluxErr', 'zApFlux', 
           'zApFluxErr', 'zApRadius', 'zKronFlux', 'zKronFluxErr', 'zKronRad', 'zExtNSigma', 'yPSFMag', 
           'yPSFMagErr', 'yApMag', 'yApMagErr', 'yKronMag', 'yKronMagErr', 'ypsfMajorFWHM', 'ypsfMinorFWHM', 
           'ymomentXX', 'ymomentXY', 'ymomentYY', 'ymomentR1', 'ymomentRH', 'yPSFFlux', 'yPSFFluxErr', 'yApFlux', 
           'yApFluxErr', 'yApRadius', 'yKronFlux', 'yKronFluxErr', 'yKronRad', 'yExtNSigma']
    columns = [x.strip() for x in columns]
    columns = [x for x in columns if x and not x.startswith('#')]
    results = ps1cone(ra,dec,radius,release='dr1',columns=columns, table="stack")
    
    tab = ascii.read(results)
    # improve the format
    
    return tab.to_pandas()













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

def geturl(ra, dec, size=240, output_size=None, filters="grizy", format="jpg", color=False, type='stack'):

    """Get URL for images in the table
    ra, dec = position in degrees
    size = extracted image size in pixels (0.25 arcsec/pixel)
    output_size = output (display) image size in pixels (default = size).
                  output_size has no effect for fits format images.
    filters = string with filters to include
    format = data format (options are "jpg", "png" or "fits")
    color = if True, creates a color image (only for jpg or png format).
            Default is return a list of URLs for single-filter grayscale images.
    Returns a string with the URL
    """

    if color and format == "fits":
        raise ValueError("color images are available only for jpg or png formats")
    if format not in ("jpg","png","fits"):
        raise ValueError("format must be one of jpg, png, fits")
    table = getimages(ra,dec,size=size,filters=filters, type=type)
    url = ("https://ps1images.stsci.edu/cgi-bin/fitscut.cgi?"
           "ra={ra}&dec={dec}&size={size}&format={format}").format(**locals())
    if output_size:
        url = url + "&output_size={}".format(output_size)
    # sort filters from red to blue
    flist = ["yzirg".find(x) for x in table['filter']]
    table = table[np.argsort(flist)]
    if color:
        if len(table) > 3:
            # pick 3 filters
            table = table[[0,len(table)//2,len(table)-1]]
        for i, param in enumerate(["red","green","blue"]):
            url = url + "&{}={}".format(param,table['filename'][i])
    else:
        urlbase = url + "&red="
        url = []
        for filename in table['filename']:
            url.append(urlbase+filename)
    return url

def getNEDInfo(df):
    df.reset_index(inplace=True, drop=True)

    df['NED_name'] = ""
    df['NED_type'] = ""
    df["NED_vel"] = np.nan
    df["NED_redshift"] = np.nan
    df["NED_mag"] = np.nan

    ra = df["raMean"]
    dec = df["decMean"]

    # setup lists for ra and dec in hr format, names of NED-identified object, and
    # separation between host in PS1 and host in NED
    ra_hms = []
    dec_dms = []
    names = []
    sep = []

    missingCounter = 0

    for index, row in df.iterrows():
        tempRA = ra[index]
        tempDEC = dec[index]
        # create a sky coordinate to query NED
        c = SkyCoord(ra=tempRA*u.degree, dec=tempDEC*u.degree, frame='icrs')
        # execute query
        result_table = []
        tempName = ""
        tempType = ""
        tempRed = np.nan
        tempVel = np.nan
        tempMag = np.nan

        try:
            result_table = Ned.query_region(c, radius=(0.00055555)*u.deg, equinox='J2000.0')
            #print(result_table)
            if len(result_table) > 0:
                missingCounter = 0
        except:
            missingCounter += 1
            #print(c)
        if len(result_table) > 0:
            result_table = result_table[result_table['Separation'] == np.min(result_table['Separation'])]
            result_table = result_table[result_table['Type'] != b'SN']
            result_table = result_table[result_table['Type'] != b'MCld']
            result_gal = result_table[result_table['Type'] == b'G']
            if len(result_gal) > 0:
                result_table = result_gal
            if len(result_table) > 0:
                result_table = result_table[result_table['Photometry Points'] == np.nanmax(result_table['Photometry Points'])]
                result_table = result_table[result_table['References'] == np.nanmax(result_table['References'])]
                #return result_table
                # NED Info is presented as:
                # No. ObjectName	RA	DEC	Type	Velocity	Redshift	Redshift Flag	Magnitude and Filter	Separation	References	Notes	Photometry Points	Positions	Redshift Points	Diameter Points	Associations
                #Split NED info up - specifically, we want to pull the type, velocity, redshift, mag
                tempNED = str(np.array(result_table)[0]).split(",")
                if len(tempNED) > 2:
                    #print("Found one!")
                    tempName = tempNED[1].strip().strip("b").strip("'")
                    if len(tempNED) > 20:
                        seps = [float(tempNED[9].strip()), float(tempNED[25].strip())]
                        if np.argmin(seps):
                            tempNED = tempNED[16:]
                    tempType =  tempNED[4].strip().strip("b").strip("''")
                    tempVel = tempNED[5].strip()
                    tempRed = tempNED[6].strip()
                    tempMag = tempNED[8].strip().strip("b").strip("''").strip(">").strip("<")
                    if tempName:
                        df.loc[index, 'NED_name'] = tempName
                    if tempType:
                        df.loc[index, 'NED_type'] = tempType
                    if tempVel:
                        df.loc[index, 'NED_vel'] = float(tempVel)
                    if tempRed:
                        df.loc[index, 'NED_redshift'] = float(tempRed)
                    #if tempMag:
                        #tempMag = re.findall(r"[-+]?\d*\.\d+|\d+", tempMag)[0]
                        #df.loc[index, 'NED_mag'] = float(tempMag)
        if missingCounter > 5000:
            print("Locked out of NED, will have to try again later...")
            return df
    return df