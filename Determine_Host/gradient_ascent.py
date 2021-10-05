# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 02:21:58 2021

@author: blgnm
"""

from astropy.wcs import WCS
import numpy.ma as ma
import warnings
from astropy.utils.exceptions import AstropyUserWarning
from astropy.utils.exceptions import AstropyWarning
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
from astropy.visualization import SqrtStretch
from scipy import interpolate
from astropy.stats import SigmaClip
from photutils import MedianBackground, MeanBackground
from photutils import Background2D
from matplotlib import colors

from panstars_query import *
from host_functions import *

import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia
def Gaia_search(ra, dec, radius):
    try:
        coord = SkyCoord(ra, dec, unit=(u.degree, u.degree), frame='icrs')
        radius = u.Quantity(radius, u.deg)
        j = Gaia.cone_search_async(coord, radius)
        r = j.get_results()
    
    
        if len(r) == 0:
            return 'Galaxy'
        else:
            
            d = r[r['dist'] == r['dist'].min()]['parallax']
            return 'Star'
    except Exception:
        
        return 'Galaxy'
    
def getSteps(hostDF):
    """
    

    Parameters
    ----------
    hostDF : Pandas DataFrame
        Pandas Data Frame with source info from panstars

    Returns
    -------
    steps : int
        How large step size should be for gradient ascent. Scales based on apparent source size.

    """
    steps = []
    
    hostRadii = hostDF['rKronRad'].values
    mean = np.nanmean(hostRadii)
    if mean == mean:
        mean = np.max([mean, 2])
        step = np.min([mean, 50])
        steps.append(step)
    else:
        steps.append(5)
    return steps

def query_ps1_noname(RA, DEC, rad):
    """
    

    Parameters
    ----------
    RA : Float
        Right Ascention of Object.
    DEC : Float
        Declenation of Object.
    rad : Float
        Radius of search in arcseconds.

    Returns
    -------
    Pandas DataFrame
        Data Frame of all sources within the radius of specified locaiton.

    """
    #print("Querying PS1 for nearest host...")
    return ps1cone(RA,DEC,rad/3600,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False)

def query_ps1_name(name, rad):
    """
    

    Parameters
    ----------
    name : str
        Objects Name to be queried.
    rad : float
        Radius of search in arcseconds.

    Returns
    -------
    Pandas DataFrame
        Data Frame of all sources within the radius of specified locaiton.

    """
    #print("Querying PS1 with host name!")
    [ra, dec] = resolve(name)
    return ps1cone(ra,dec,rad/3600,table="stack",release="dr1",format="csv",columns=None,baseurl="https://catalogs.mast.stsci.edu/api/v0.1/panstarrs", verbose=False)



def get_clean_img(ra, dec, px, band):
    image_data_mask = get_PS1_mask(ra, dec, int(px), band).data
    image_data_num = get_PS1_type(ra, dec, int(px), band, 'stack.num').data
    image_data = get_PS1_Pic(ra, dec, int(px), band)
    hdu = image_data
    image_data = image_data.data
    
    wcs = WCS(hdu.header)
    
    bit = image_data_mask
    mask = image_data_mask
    
    for i in np.arange(np.shape(bit)[0]):
            for j in np.arange(np.shape(bit)[1]):
                if image_data_mask[i][j] == image_data_mask[i][j]:
                    bit[i][j] = "{0:016b}".format(int(image_data_mask[i][j]))
                    tempBit = str(bit[i][j])[:-2]
                    if len(str(int(bit[i][j]))) > 12:
                        if (tempBit[-6] == 1) or (tempBit[-13] == 1):
                            mask[i][j] = np.nan
                    elif len(str(int(bit[i][j]))) > 5:
                        if (tempBit[-6] == 1):
                            mask[i][j] = np.nan

    mask = ~np.isnan(image_data_mask)
    mask_num = image_data_num
    #weighted
    #image_data *= image_data_wt
    image_masked = ma.masked_array(image_data, mask=mask)
    image_masked_num = ma.masked_array(image_masked, mask=mask_num)
    
    return np.array(image_masked), wcs, hdu

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

def updateStep(px, gradx, grady, step, point, size):
    max_x = px
    max_y =  px
    grad = np.array([gradx[point[0], point[1]], grady[point[0], point[1]]])
    #make sure we move at least one unit in grid spacing - so the grad must have len 1
#    if grad[0] + grad[1] > 0:
    ds = step/np.sqrt(grad[0]**2 + grad[1]**2)
    ds = np.nanmin([ds, step])
#    else:
#        ds = step

    newPoint = [point[0] + ds*grad[0], point[1] + ds*grad[1]]
    newPoint = [int(newPoint[0]), int(newPoint[1])] #round to nearest index
    if (newPoint[0] >= max_x) or (newPoint[1] >= max_y) or (newPoint[0] < 0) or (newPoint[1] < 0):
        #if we're going to go out of bounds, don't move
        return point
    elif ((newPoint == point) and (size == 'large')): #if we're stuck, perturb one pixel in a random direction:
        a = np.random.choice([-1, 0, 1], 2)#
        newPoint = [newPoint[0] + a[0], newPoint[1] + a[1]]
    return newPoint

def dist(p1, p2):
    """
    

    Parameters
    ----------
    p1 : float
        Location 1.
    p2 : float
        location 2.

    Returns
    -------
    float
        Distance from location 1 to location 2.

    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)    
    
def get_PS1_Pic(ra, dec, rad, band, safe=False):
    """
    

    Parameters
    ----------
    ra : Float
        Right ascension of object.
    dec : Float
        Declanation of object.
    rad : Float
        Radius in arcseconds from object to search.
    band : str
        Filter image should be in.
    safe : Bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    Numpy Array
        Image of radius rad at the specified location from the panstars survey.

    """
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits")
    fh = fits.open(fitsurl[0])
    return fh[0]
def get_PS1_type(ra, dec, rad, band, type):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits", type=type)
    fh = fits.open(fitsurl[0])
    return fh[0]

def get_PS1_wt(ra, dec, rad, band):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits", type='stack.wt')
    fh = fits.open(fitsurl[0])
    return fh[0]
def get_PS1_mask(ra, dec, rad, band):
    fitsurl = geturl(ra, dec, size=rad, filters="{}".format(band), format="fits", type='stack.mask')
    fh = fits.open(fitsurl[0])
    return fh[0]



def gradient_ascent(hostDF, ra, dec, plot = False):
    """
    

    Parameters
    ----------
    hostDF : Pandas DataFrame
        DataFrame of sources panstars features
    SN_ra : Float
        Right Ascension of supernovae.
    SN_dec : Float
        Declination of supernovae.
    plot : Bool, optional
        Choose whether to plot gradient ascent. The default is False.

    Returns
    -------
    Host DataFrame with updated location.

    """
    warnings.filterwarnings('ignore', category=AstropyUserWarning)
    warnings.filterwarnings('ignore', category=AstropyWarning)
    
    step_sizes = getSteps(hostDF)
    unchanged = []
    N_associated = 0
    
    px = 800
    
    g_img, wcs, g_hdu  = get_clean_img(ra, dec, px, 'g')
    g_mask = np.ma.masked_invalid(g_img).mask
    r_img, wcs, r_hdu  = get_clean_img(ra, dec, px, 'r')
    r_mask = np.ma.masked_invalid(r_img).mask
    i_img, wcs, i_hdu  = get_clean_img(ra, dec, px, 'i')
    i_mask = np.ma.masked_invalid(i_img).mask
    
    
    nancount = 0
    obj_interp = []
    
    for obj in [g_img, r_img, i_img]:
        data = obj
        mean, median, std = sigma_clipped_stats(data, sigma=20.0)
        daofind = DAOStarFinder(fwhm=3.0, threshold=20.*std)
        sources = daofind(data - median)
        
        try:
            xvals = np.array(sources['xcentroid'])
            yvals = np.array(sources['ycentroid'])

            for k in np.arange(len(xvals)):
                tempx = xvals[k]
                tempy = yvals[k]
                yleft = np.max([int(tempy) - 7, 0])
                yright = np.min([int(tempy) + 7, np.shape(data)[1]-1])
                xleft = np.max([int(tempx) - 7, 0])
                xright = np.min([int(tempx) + 7, np.shape(data)[1]-1])

                for r in np.arange(yleft,yright+1):
                    for j in np.arange(xleft, xright+1):
                        if dist([xvals[k], yvals[k]], [j, r]) < 5:
                            data[r, j] = np.nan
            nancount += np.sum(np.isnan(data))
            positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
            apertures = CircularAperture(positions, r=5.)
            norm = ImageNormalize(stretch=SqrtStretch())
            
            if plot == True:
                fig = plt.figure(figsize=(10,10))
                ax = fig.gca()
                ax.imshow(data)
                apertures.plot(color='blue', lw=1.5, alpha=0.5)
                plt.axis('off')
                plt.show()
                plt.close()
            
        except:
            print('No stars here!')
            
        backx = np.arange(0,data.shape[1])
        backy = np.arange(0, data.shape[0])
        backxx, backyy = np.meshgrid(backx, backy)
        #mask invalid values
        array = np.ma.masked_invalid(data)
        x1 = backxx[~array.mask]
        y1 = backyy[~array.mask]
        newarr = array[~array.mask]

        data = interpolate.griddata((x1, y1), newarr.ravel(), (backxx, backyy), method='cubic')
        obj_interp.append(data)
        
    #gvar = np.var(obj_interp[0])
    #gmean = np.nanmedian(obj_interp[0])
    gMax = np.nanmax(obj_interp[0])

    g_ZP = g_hdu.header['ZPT_0001']
    r_ZP = r_hdu.header['ZPT_0001']
    i_ZP = i_hdu.header['ZPT_0001']

    #combining into a mean img -
    # m = -2.5*log10(F) + ZP
    
    gmag = -2.5*np.log10(obj_interp[0]) + g_ZP
    rmag = -2.5*np.log10(obj_interp[1]) + r_ZP
    imag = -2.5*np.log10(obj_interp[2]) + i_ZP

    #now the mean can be taken
    mean_zp = (g_ZP + r_ZP + i_ZP)/3
    meanMag = (gmag + rmag + imag)/3
    meanImg = 10**((mean_zp-meanMag)/2.5) #convert back to flux

    #meanImg = (obj_interp[0] + obj_interp[0] + obj_interp[0])/3
    #print("NanCount = %i"%nancount,file=f)
    #mean_center = np.nanmean([g_img[int(px/2),int(px/2)], i_img[int(px/2),int(px/2)], i_img[int(px/2),int(px/2)]])
    #if mean_center != mean_center:
    #    mean_center = 1.e-30
    
    mean_center = meanImg[int(px/2),int(px/2)]
    
    #print("Mean_center = %f" % mean_center,file=f)
    #mean, median, std = sigma_clipped_stats(meanImg, sigma=10.0)
    meanImg[meanImg != meanImg] = 1.e-30
    mean, median, std = sigma_clipped_stats(meanImg, sigma=10.0)
    #print("mean image = %e"% mean, file=f)
    aboveCount = np.sum(meanImg > 1.)
    aboveCount2 = np.sum(meanImg[int(px/2)-100:int(px/2)+100, int(px/2)-100:int(px/2)+100] > 1.)
    aboveFrac2= aboveCount2/40000
    #print("aboveCount = %f"% aboveCount,file=f)
    #print("aboveCount2 = %f "% aboveCount2, file=f)
    totalPx = px**2
    aboveFrac = aboveCount/totalPx
    #print("aboveFrac= %f" % aboveFrac, file=f)
    #print("aboveFrac2 = %f "% aboveFrac2, file=f)
        
    if ((median <15) and (np.round(aboveFrac2, 2) < 0.70)) or ((mean_center > 1.e3) and (np.round(aboveFrac,2) < 0.60) and (np.round(aboveFrac2,2) < 0.75)):
        bs = 15
        fs = 1
        if aboveFrac2 < 0.7:
            step_sizes = 2.
        else:
            step_sizes = 10.
        #print("Small filter", file=f)
        size = 'small'
    elif ((mean_center > 40) and (median > 500) and (aboveFrac > 0.60)) or ((mean_center > 300) and (aboveFrac2 > 0.7)):
        bs = 75 #the big sources
        fs = 3
        #print("Large filter", file=f)
        print(step_sizes)
        step_sizes = np.max([step_sizes[0], 50])
        size = 'large'
        #if step_sizes[int(i)] == 5:
        #    step_sizes[int(i)] *= 5
        #    step_sizes[int(i)] = np.min([step_sizes[int(i)], 50])
        #if mean_center < 200: #far from the center with a large host
        #    fs = 5
        #elif mean_center < 5000:
        #    step_sizes[int(i)] = np.max([step_sizes[int(i)], 50])
        #size = 'large'
    else:
        bs = 40 #everything in between
        fs = 3
        #print("Medium filter", file=f)
        #if step_sizes[int(i)] == 5:
        #    step_sizes[int(i)] *= 3
#        step_sizes[int(i)] = np.max([step_sizes[int(i)], 25])
        step_sizes = np.max([step_sizes[0], 15])
        size = 'medium'
    #    step_sizes[int(i)] *= 3
    
    #if (median)
    sigma_clip = SigmaClip(sigma=15.)
    bkg_estimator = MeanBackground()
    #bkg_estimator = BiweightLocationBackground()
    bkg3_g = Background2D(g_img, box_size=bs, filter_size=fs,
     sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkg3_r = Background2D(r_img, box_size=bs, filter_size=fs,
     sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    bkg3_i = Background2D(i_img, box_size=bs, filter_size=fs,
     sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    #pretend the background is in counts too (I think it is, right?) and average in mags
    bkg3_g.background[bkg3_g.background < 0] = 1.e-30
    bkg3_r.background[bkg3_r.background < 0] = 1.e-30
    bkg3_i.background[bkg3_i.background < 0] = 1.e-30

    backmag_g = -2.5*np.log10(bkg3_g.background) + g_ZP
    backmag_r = -2.5*np.log10(bkg3_r.background) + r_ZP
    backmag_i = -2.5*np.log10(bkg3_i.background) + i_ZP

    mean_zp = (g_ZP + r_ZP + i_ZP)/3.
    backmag = 0.333*backmag_g + 0.333*backmag_r + 0.333*backmag_i
    background = 10**(mean_zp-backmag/2.5)

    if plot == True:
        fig, axs = plt.subplots(1, 3, sharex=True, sharey=True,figsize=(20,10))
        axs[0].imshow(bkg3_g.background)
        axs[0].axis('off')
        axs[1].imshow(bkg3_r.background)
        axs[1].axis('off')
        axs[2].imshow(bkg3_i.background)
        axs[2].axis('off')
        plt.show()
        plt.close()
    
        
    mean, median, std = sigma_clipped_stats(meanImg, sigma=1.0)
    meanImg[meanImg <= (mean)] = 1.e-30
    meanImg[meanImg < 0] = 1.e-30

    if plot:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.imshow((meanImg)/np.nanmax(meanImg))
        plt.axis('off')
        #plt.savefig("quiverMaps/normalizedMeanImage_%s.png" % transient_name, bbox_inches='tight')
        plt.show()
        plt.close()

        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.imshow(background/np.nanmax(background))
        plt.axis('off')
        #plt.savefig("quiverMaps/normalizedMeanBackground_%s.png" % transient_name, bbox_inches='tight')
        plt.show()
        plt.close()

    if nancount > 1.e5:
        imgWeight = 0
    elif (mean_center > 1.e4): #and (size is not 'large'):
        imgWeight = 0.75
    elif size == 'medium':
        imgWeight = 0.33
    else:
        imgWeight = 0.10
    #print("imgWeight= %f"%imgWeight, file=f)
    fullbackground = ((1-imgWeight)*background/np.nanmax(background) + imgWeight*meanImg/np.nanmax(meanImg))*np.nanmax(background)
#            background  = (0.66*background/np.max(background) +  imgWeight*meanImg/np.nanmax(meanImg))*np.max(background)
    n = px
    X, Y = np.mgrid[0:n, 0:n]
    dx, dy = np.gradient(fullbackground.T)

    n_plot = 10

    dx_small = dx[::n_plot, ::n_plot]
    dy_small = dy[::n_plot, ::n_plot]
    #print("step = %f"%  step_sizes[int(i)], file=f)

    start = [[int(px/2),int(px/2)]] #the center of the grid

    if True:
    #if background[int(px/2),int(px/2)] > 0: #if we have some background flux (greater than 3 stdevs away from the median background), follow the gradient
        start.append(updateStep(px, dx, dy, step_sizes, start[-1], size))
        for j in np.arange(1.e3):
            start.append(updateStep(px, dx, dy, step_sizes, start[-1], size))
        it_array = np.array(start)
        endPoint = start[-1]

        if plot:
            fig  = plt.figure(figsize=(10,10))
            ax = fig.gca()
            ax.imshow(fullbackground)
            plt.axis("off")
            plt.show()
            #plt.savefig("quiverMaps/fullBackground_%s.png"%transient_name, bbox_inches='tight')
            plt.close()

        coords = wcs.wcs_pix2world(endPoint[0], endPoint[1], 0., ra_dec_order = True) # Note the third argument, set to 0, which indicates whether the pixel coordinates should be treated as starting from (1, 1) (as FITS files do) or from (0, 0)
        #print("Final ra, dec after GD : %f %f"% (coords[0], coords[1]), file=f)
        col = '#D34E24'
        col2 = '#B54A24'
        #lookup by ra, dec
        try:
            if size == 'large':
                a = query_ps1_noname(float(coords[0]), float(coords[1]), 20)
            else:
                a = query_ps1_noname(float(coords[0]), float(coords[1]), 5)
        except TypeError:
             print('hi')
             
        if a:
            #print("Found a host here!", file=f)
            a = ascii.read(a)
            a = a.to_pandas()
    
            a = a[a['nDetections'] > 1]
            #a = a[a['ng'] > 1]
            #a = a[a['primaryDetection'] == 1]
            smallType = ['AbLS', 'EmLS' , 'EmObj', 'G', 'GammaS', 'GClstr', 'GGroup', 'GPair', 'GTrpl', 'G_Lens', 'IrS', 'PofG', 'RadioS', 'UvES', 'UvS', 'XrayS', '', 'QSO', 'QGroup', 'Q_Lens']
            medType = ['G', 'IrS', 'PofG', 'RadioS', 'GPair', 'GGroup', 'GClstr', 'EmLS', 'RadioS', 'UvS', 'UvES', '']
            largeType = ['G', 'PofG', 'GPair', 'GGroup', 'GClstr']
            if len(a) > 0:
                a = getNEDInfo(a)
                if (size == 'large'):# and (np.nanmax(a['rKronRad'].values) > 5)):
                #    print("L: picking the largest >5 kronRad host within 10 arcsec", file=f)
                    #print("L: picking the closest NED galaxy within 20 arcsec", file=f)
                    #a = a[a['rKronRad'] == np.nanmax(a['rKronRad'].values)]
                    tempA = a[a['NED_type'].isin(largeType)]
                    if len(tempA) > 0:
                        a = tempA
                    tempA = a[a['NED_type'] == 'G']
                    if len(tempA) > 0:
                        a = tempA
                    #tempA = a[a['NED_mag'] == np.nanmin(a['NED_mag'])]
                    #if len(tempA) > 0:
                    #    a = tempA
                    if len(a) > 1:
                        a = a.iloc[[0]]
                elif (size == 'medium'):
                    #print("M: Picking the largest host within 5 arcsec", file=f)
                    #print("M: Picking the closest NED galaxy within 5 arcsec", file=f)
                    #a = a[a['rKronRad'] == np.nanmax(a['rKronRad'].values)]
                    tempA = a[a['NED_type'].isin(medType)]
                    if len(tempA) > 0:
                        a = tempA
                    if len(a) > 1:
                        a = a.iloc[[0]]
                else:
                    tempA = a[a['NED_type'].isin(smallType)]
                    if len(tempA) > 0:
                        a = tempA
                    a = a.iloc[[0]]
                    #print("S: Picking the closest non-stellar source within 5 arcsec", file=f)
                #else:
                #    f.flush()
                #    continue
                #threshold = [1, 1, 0, 0, 0, 0]
                #flag = ['nDetections', 'nr', 'rPlateScale', 'primaryDetection', 'rKronRad', 'rKronFlux']
                #j = 0
                #while len(a) > 1:
                #    if np.sum(a[flag[int(j)]] > threshold[int(j)]) > 0:
                #        tempA = a[a[flag[int(j)]] > threshold[int(j)]]
                #        j += 1
                #        a = tempA
                #        if (j == 6):
                #            break
                #    else:
                #        break
                #if len(a) > 1:
                #    if len(~a['rKronRad'].isnull()) > 0:
                #        a = a[a['rKronRad'] == np.nanmax(a['rKronRad'].values)]
                #    else:
                #        a = a.iloc[0]
                #print("Nice! Host association chosen.", file=f)
                #print("NED type: %s" % a['NED_type'].values[0], file=f)
                #print(a['objID'].values[0], file=f)
                #print("Chosen Host RA and DEC: %f %f"% (a['raMean'], a['decMean']), file=f)
                #SN_dict_postDLR[transient_name] = a['objID'].values[0]
                #print("Dict value: %i"%SN_dict_postDLR[transient_name],file=f)
                N = len(hostDF)
                hostDF = pd.concat([hostDF, a], ignore_index=True)
                N2 = len(hostDF)
                if N2 != (N+1):
                    #print("ERROR! Value not concatenated!!", file=f)
                    return
                finalRA = np.array(a['raMean'])
                finalDEC = np.array(a['decMean'])
                col = 'tab:green'
                col2 = '#078840'     
            else:
                unchanged.append(1)
        else:
            unchanged.append(1)
        if plot:
            fig = plt.figure(figsize=(20,20))
            ax = fig.gca()
            ax.imshow(i_img, norm=colors.PowerNorm(gamma = 0.5, vmin=1, vmax=1.e4), cmap='gray')#, cmap='gray', norm=LogNorm())
            it_array = np.array(start)
            ax.plot(it_array.T[0], it_array.T[1], "--", lw=5, c=col, zorder=20)
            ax.scatter([int(px/2)], [int(px/2)], marker='*', s=1000, color='#f3a712', zorder=50)
            ax.scatter(endPoint[0], endPoint[1], marker='*', lw=4, s=1000, facecolor='#f3a712', edgecolor=col2, zorder=200)
            ax.quiver(X[::n_plot,::n_plot], Y[::n_plot,::n_plot], dx[::n_plot,::n_plot], dy[::n_plot,::n_plot], color='#845C9B', angles='xy', scale_units = 'xy')
            plt.axis('off')
            plt.show()
            #plt.savefig("quiverMaps/quiverMap_%s.png"%transient_name, bbox_inches='tight')
            plt.close()
             
        N_associated += 1
    
    
    
    
    return hostDF, finalRA, finalDEC
             
             
             
             