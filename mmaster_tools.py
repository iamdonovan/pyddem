from __future__ import print_function
from future_builtins import zip
from functools import partial
import os
import shutil
from glob import glob
import pyproj
import numpy as np
import numpy.polynomial.polynomial as poly
import gdal
import ogr
import fiona
import fiona.crs
import matplotlib.pylab as plt
import pandas as pd
import scipy.optimize as optimize
from matplotlib.backends.backend_pdf import PdfPages
from shapely.geometry.polygon import Polygon, orient
from shapely.geometry import mapping, LineString, Point
from shapely.ops import cascaded_union, transform
from pybob.coreg_tools import dem_coregistration
from pybob.coreg_tools import false_hillshade
from pybob.coreg_tools import get_slope
from pybob.coreg_tools import final_histogram
from pybob.GeoImg import GeoImg
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython import embed

def make_mask(inpoly, pts, raster_out=False):
    """
    Create a True/False mask to determine whether input points are within a polygon.
    
    Parameters
    ----------
    inpoly : shapely Polygon
        Input polygon to use to create mask.
    pts : array-like or GeoImg
        Either a list of (x,y) coordinates, or a GeoImg. If a list of (x,y) coordinates,
        make_mask uses shapely.polygon.contains to determine if the point is within inpoly.
        If pts is a GeoImg, make_mask uses ogr and gdal to "rasterize" inpoly to the GeoImg raster.
    raster_out : bool
        Kind of output to return. If True, pts must be a GeoImg.
        
    Returns
    -------
    mask : array-like, bool
        An array which is True where pts is within inpoly, False elsewhere.
    """
    if not raster_out:
        mask = []
        for pt in pts:
            mask.append(inpoly.contains(Point(pt)))
        return mask
    else:
        if type(pts) is not GeoImg:
            raise TypeError('raster_out is requested, but pts is not a GeoImg')
        # use gdal, ogr
        # first, create a layer for the polygon to live in.
        drv = ogr.GetDriverByName('Memory')
        dst_ds = drv.CreateDataSource('out') 
        masklayer = dst_ds.CreateLayer('test', geom_type=ogr.wkbMultiPolygon, srs=pts.spatialReference)
        feature = ogr.Feature(masklayer.GetLayerDefn())
        feature.SetGeometry(ogr.CreateGeometryFromWkt(inpoly.crs_wkt))
        masklayer.CreateFeature(feature)
        feature.Destroy()
        # now, create the raster to burn the mask to.
        masktarget = gdal.GetDriverByName('MEM').Create('', pts.npix_x, pts.npix_y, 1, gdal.GDT_Byte)
        masktarget.SetGeoTransform((pts.xmin, pts.dx, 0, pts.ymax, 0, pts.dy))
        masktarget.SetProjection(pts.proj)
        masktarget.GetRasterBand(1).Fill(0)
        gdal.RasterizeLayer(masktarget, [1], masklayer)
        mask = masktarget.GetRasterBand(1).ReadAsArray()
        mask[mask != 0] = 1
        return mask == 1


def make_group_id(inmat, grpid):
    """
    Make a unique ID for statistical analysis.
    
    Parameters
    ----------
    inmat : array-like
        Input matrix to create ID for.
    grpid : array-like
        Matrix of input group IDs.

    Returns
    -------
    outmat : array-like
        Output of group IDs.
    """
    return np.floor(inmat / grpid) * grpid

def get_group_statistics(invar,indata,indist=500):
    
    xxid=make_group_id(invar,indist) # across track coordinates (grouped 500m)
    #yyid=make_group_id(yyr,500) # along track coordinates (grouped 500m)

    # Calculate group statistics 
    mykeep=np.isfinite(indata)
    data = pd.DataFrame({'dH': indata[mykeep],'XX': xxid[mykeep]})
    xxgrp=data['dH'].groupby(data['XX']).describe()
    
    return xxgrp

def reproject_geometry(src_data, src_epsg, dst_epsg):
    """
    Reprojects src_data from one coordinate system to another.
    
    Parameters
    ----------
    src_data : geometry object
        shapely geometry object to be reprojected.
    src_epsg : int or str
        EPSG code for source CRS
    dst_epsg : int or str
        EPSG code for destination CRS
        
    Returns
    -------
    dst_data : geometry object
        reprojected shapely geometry object.    
    """
    src_proj = pyproj.Proj(init='epsg:{}'.format(src_epsg))
    dst_proj = pyproj.Proj(init='epsg:{}'.format(dst_epsg))
    project = partial(pyproj.transform, src_proj, dst_proj)
    return transform(project, src_data)


def nmad(data):
    """
    Calculate the Normalized Median Absolute Deviation (NMAD) of the input dataset.
    
    Parameters
    ----------
    data : array-like
        Dataset on which to calculate nmad
    
    Returns
    -------
    nmad : array-like
        Normalized Median Absolute Deviation of input dataset
    """
    return np.nanmedian(np.abs(data) - np.nanmedian(data))


def get_aster_footprint(gran_name, epsg='4326'):
    """
    Create shapefile of ASTER footprint from .met file.
    
    Parameters
    ----------
    gran_name : str
        ASTER granule name to use; assumed to also be the folder in which .met file(s) are stored.
    epsg : str or int
        EPSG code for coordinate system to use.
    Returns
    -------
    footprint : Polygon
        shapely Polygon representing ASTER footprint, re-projected to given geometry.
    """
    metlist = glob(os.path.sep.join([os.path.abspath(gran_name), '*.met']))
    
    schema = {'properties': [('id', 'int')], 'geometry': 'Polygon'}
    outshape = fiona.open(os.path.sep.join([os.path.abspath(gran_name), gran_name + '_Footprint.shp']), 'w',
                          crs=fiona.crs.from_epsg(int(epsg)), driver='ESRI Shapefile', schema=schema)    

    footprints = []
    for m in metlist:
        clean = [line.strip() for line in open(m).read().split('\n')]

        if os.path.sep in m:
            m = m.split(os.path.sep)[-1]

        latinds = [i for i, line in enumerate(clean) if 'GRingPointLatitude' in line]
        loninds = [i for i, line in enumerate(clean) if 'GRingPointLongitude' in line]

        latlines = clean[latinds[0]:latinds[1]+1]
        lonlines = clean[loninds[0]:loninds[1]+1]

        lonvalstr = lonlines[2]
        latvalstr = latlines[2]

        lats = [float(val) for val in latvalstr.strip('VALUE =()').split(',')]
        lons = [float(val) for val in lonvalstr.strip('VALUE =()').split(',')]

        coords = list(zip(lons, lats))
        footprints.append(Polygon(coords))

    footprint = cascaded_union(footprints)
    footprint = footprint.simplify(0.0001)
    outprint = reproject_geometry(footprint, 4326, epsg)
    outshape.write({'properties': {'id': 1}, 'geometry': mapping(outprint)})
    outshape.close()
    return outprint


def orient_footprint(fprint):
    """
    Orient ASTER footprint coordinates to be clockwise, with upper left coordinate first.
    
    Parameters
    ----------
    fprint : Polygon
        Footprint polygon to orient.
    
    Returns
    -------
    o_fprint : Polygon
        Re-oriented copy of input footprint.
    """
    # orient the footprint coordinates so that they are clockwise
    fprint = orient(fprint, sign=-1)
    x, y = fprint.boundary.coords.xy
    x = x[:-1] # drop the last coordinate, which is a duplicate of the first
    y = y[:-1]
    # as long as the footprints are coming from the .met file, the upper left corner 
    # will be the maximum y value.
    upper_left = np.argmax(y)
    new_inds = range(upper_left, len(x)) + range(0, upper_left)
    return Polygon(list(zip(np.array(x)[new_inds], np.array(y)[new_inds])))


def get_track_angle(fprint, track_dist):
    """
    Calculate the angle made by the ASTER flight track.
    
    Parameters
    ----------
    fprint : shapely Polygon
        Footprint of ASTER scene.
    track_dist : float
        Distance along flight track within scene to calculate angle.
        
    Returns
    -------
    track_angle : float
        Angle, in degrees, of flight track.
    """
    # orient the footprint
    fprint = orient_footprint(fprint)
    x, y = fprint.boundary.coords.xy    
    # upper_left = np.argmax(y)
    upper_right = np.argmax(x)
    lower_right = np.argmin(y)
    lower_left = np.argmin(x)
    
    lside = range(len(x)-1, lower_left-1, -1)
    rside = range(upper_right, lower_right+1)
    
    left_side = LineString(list(zip(np.array(x)[lside], np.array(y)[lside])))
    right_side = LineString(list(zip(np.array(x)[rside], np.array(y)[rside])))
    
    lproj = left_side.interpolate(track_dist)
    rproj = right_side.interpolate(track_dist)    
    # get the angle of the line formed by connecting lproj, rproj
    dx = lproj.x - rproj.x
    dy = lproj.y - rproj.y    
    
    return 90 + np.rad2deg(np.arctan(dx/dy))

def preprocess(mst_dem, slv_dem, glac_mask=None, land_mask=None, cwd='.'):
    """
    Pre-process ASTER scene to enable cross- and along-track corrections. Co-registers the
    ASTER (slave) and external (master) DEMs, and shifts the orthoimage and correlation mask
    based on the offset calculated.   
    
    Parameters
    ----------
    mst_dem : string or GeoImg
        Path to filename or GeoImg dataset representing external "master" DEM.
    slv_dem : string or GeoImg
        Path to filename or GeoImg dataset representing ASTER DEM.
    glac_mask : string, optional
        Path to shapefile representing points to exclude from co-registration
        consideration (i.e., glaciers).
    land_mask : string, optional
        Path to shapefile representing points to include in co-registration
        consideration (i.e., stable ground/land).
    cwd : string, optional
        Working directory to use [Assumes '.'].
        
    Returns
    -------
    mst_coreg : GeoImg
        GeoImg dataset represeting co-registered external "master" DEM.
    slv_coreg : GeoImg
        GeoImg dataset representing co-registered ASTER "slave" DEM.
    shift_params : tuple
        Tuple containing x, y, and z shifts calculated during co-regisration process.
    """
    
    # filter DEM using correlation mask !!! NEED to update to allow for either GeoIMG or pathname as INPUT
    ast_name=slv_dem.rsplit('_Z.tif')[0]
    mask_name=os.path.sep.join([cwd, '{}_CORR.tif'.format(ast_name)])
    slv_name=os.path.sep.join([cwd, '{}_filtZ.tif'.format(ast_name)])
    mask_raster_threshold(slv_dem,mask_name,slv_name,50,np.float32)
        
    # co-register master, slave
    out_dir = os.path.sep.join([cwd, 'coreg'])
    mst_coreg, slv_coreg, shift_params = dem_coregistration(mst_dem, slv_name, glaciermask=glac_mask, 
                                                            landmask=land_mask, outdir=out_dir)
    print(slv_coreg.filename)
    # remove coreg folder, save slv as *Zadj1.tif, but save output.pdf
    shutil.move(os.path.sep.join([out_dir, 'CoRegistration_Results.pdf']), os.path.sep.join([cwd, 'CoRegistration_Results.pdf']))
    shutil.move(os.path.sep.join([out_dir, 'coreg_params.txt']), os.path.sep.join([cwd, 'coreg_params.txt']))
    shutil.rmtree(out_dir)
    slv_coreg.write(os.path.sep.join([cwd, '{}_Zadj1.tif'.format(ast_name)]))
    # shift ortho, corr masks, save as *adj1.tif
    ortho = GeoImg(ast_name + '_V123.tif')
    corr = GeoImg(ast_name + '_CORR.tif')
    
    ortho.shift(shift_params[0], shift_params[1])
    ortho.write(os.path.sep.join([cwd, '{}_V123_adj1.tif'.format(ast_name)]))
    
    corr.shift(shift_params[0], shift_params[1])
    corr.write(os.path.sep.join([cwd, '{}_CORR_adj1.tif'.format(ast_name)]), dtype=np.uint8)  
    
    plt.close("all")                                                      
    return mst_coreg, slv_coreg, shift_params

def mask_raster_threshold(rasname,maskname,outfilename,threshold=50,datatype=np.float32):
    '''
    filters values in rasname if maskname is less then threshold
    '''
    myras= GeoImg(rasname)
    mymask = GeoImg(maskname)
    
    rem = mymask.img < threshold
    myras.img[rem] = np.nan
    myras.write(outfilename,driver='GTiff', dtype=datatype)
    
    return myras

def RMSE(indata): 
    """ Return root mean square of indata."""
    myrmse = np.sqrt(np.nanmean(indata**2))
    return myrmse

def calculate_dH(mst_dem,slv_dem):
    # mst_dem and slv_dem must be GeoIMG objects with same pixels
    # 1) calculates the differences
    # 2) Runs a 5x5 pixel median filter
    # 3) Hard Removes values > 40
    
    dH = mst_dem.img-slv_dem.img
    dH2 = mst_dem.copy(new_raster=(ndimage.median_filter(dH,5)))
    #dH = master.copy(new_raster=(master.img-slave.img))
    fmask = np.abs(dH2.img) > 40
    fmask.astype(np.int)
    dH2.img[fmask] = np.nan
    
    plt.figure(figsize=(5,5))
    plt.imshow(dH2.img,interpolation='nearest',cmap='gray')
    plt.show()
    
    return dH2

def get_xy_rot(mst_dem,myang):
    # creates matrices for along and across track distances from a reference dem and a raster angle map (in radians)
    
    xx, yy = mst_dem.xy(grid=True)
    xx = xx - np.min(xx)
    yy = yy - np.min(yy)
    #xxr = np.multiply(xx,np.rad2deg(np.cos(myang))) + np.multiply(-1*yy,np.rad2deg(np.sin(myang)))
    #yyr = np.multiply(xx,np.rad2deg(np.sin(myang))) + np.multiply(yy,np.rad2deg(np.cos(myang)))    
    xxr = np.multiply(xx,np.cos(myang)) + np.multiply(-1*yy,np.sin(myang))
    yyr = np.multiply(xx,np.sin(myang)) + np.multiply(yy,np.cos(myang))    

    # TO USE FOR INITIALIZING START AT ZERO
    xxr = xxr - np.nanmin(xxr)
    yyr = yyr - np.nanmin(yyr)
    
    plt.figure(figsize=(5,5))
    plt.imshow(xxr,interpolation='nearest')
    plt.show()

    plt.figure(figsize=(5,5))
    plt.imshow(yyr,interpolation='nearest')
    plt.show()

    return xxr, yyr

def get_atrack_coord(mst_dem,myangN,myangB):
    # creates matrices for along and across track distances from a reference dem and a raster angle map (in radians)
    
    xx, yy = mst_dem.xy(grid=True)
    xx = xx - np.min(xx)
    yy = yy - np.min(yy)
    #xxr = np.multiply(xx,np.rad2deg(np.cos(myang))) + np.multiply(-1*yy,np.rad2deg(np.sin(myang)))
    #yyr = np.multiply(xx,np.rad2deg(np.sin(myang))) + np.multiply(yy,np.rad2deg(np.cos(myang)))    
    #xxr = np.multiply(xx,np.cos(myang)) + np.multiply(-1*yy,np.sin(myang))

    yyn = np.multiply(xx,np.sin(myangN)) + np.multiply(yy,np.cos(myangN))    
    yyb = np.multiply(xx,np.sin(myangB)) + np.multiply(yy,np.cos(myangB))    

   
    plt.figure(figsize=(5,5))
    plt.imshow(yyn,interpolation='nearest')
    plt.show()

    plt.figure(figsize=(5,5))
    plt.imshow(yyb,interpolation='nearest')
    plt.show()

    return yyn, yyb

def polynomial_fit(x,y):
    
    
    #edge removal
    x = x[5:-5]
    y = y[5:-5]     
    
    plt.figure(figsize=(7,5))
    plt.plot(x,y,'.')
    rmse=np.empty(5)
    coeffs = np.zeros((5,7))
    xnew = np.arange(np.min(x),np.max(x),1000)
    for deg in np.arange(2,7):
        #print(deg)
        p,r = poly.polyfit(x,y,deg,full=True)
        p2 = poly.polyval(xnew, p)
        plt.plot(xnew,p2)
        coeffs[deg-2,0:p.size] = p
        rmse[deg-2] = np.sqrt(np.divide(r[0],y.size-deg))
    
    plt.figure(figsize=(7,5))
    plt.plot(np.arange(2,7),rmse)
    
    # Choose order of polynomial - 2 options 
    # a) lowest RMSE or 
    # b) by checking the percent improvemnt (more robust?)
    
    # a) lowest RMSE
    fidx = rmse.argmin()    

    # b) [DEFAULT??] find the highest order which gave a 5% improvemnt in the RMSE
    perimp=np.divide(rmse[:-1]-rmse[1:],rmse[:-1])*100
    #print(perimp)
    #print(rmse)
    fmin=np.asarray(np.where(perimp>5))
    if fmin.size!=0:
        fidx = fmin[0,-1]+1
    # print('fidx: {}'.format(fidx))
    
    print("Polynomial Order Selected: ",fidx+2)
    return coeffs[fidx], rmse[fidx]

def fitfun_sumofsin(xx,p): 
    myval=0
    for bb in np.arange(0,p.size-1,3):
        myval=myval + p[bb]*np.sin(p[bb+1]*xx+p[bb+2])
    return myval

def function_sum_of_sin(xx,yy,lb,ub,pp,ylim=None):

    def errfun(p, xx, yy): return fitfun_sumofsin(xx,p) - yy

    fig = plt.figure(figsize=(7, 5), dpi=200)
    #fig.suptitle(title, fontsize=14)
    plt.plot(xx, yy, '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')
    
    mykeep=np.isfinite(yy)
    xx=xx[mykeep]
    yy=yy[mykeep]
    if xx.size > 5000:
        mysamp = np.random.randint(0, xx.size, 5000)
    else:
        mysamp = np.arange(0, xx.size)
    
    myorder=3
    mycost=np.zeros(myorder)
    for order in np.arange(1,myorder+1):
        print(order)
        myrow=order-1
        #lb = np.asarray([3, np.divide(2*np.pi,60000), -30])
        #ub = [30, np.divide(2*np.pi,6000), 30]
        lbb = np.squeeze(np.matlib.repmat(lb,1,order))
        ubb = np.squeeze(np.matlib.repmat(ub,1,order))
        p0 = np.divide(lbb+ubb,2)
        #p1, success, _ = optimize.least_squares(errfun, p0[:], args=([xdata], [ydata]), method='trf', bounds=([lb],[ub]), loss='soft_l1', f_scale=0.1)
        #myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5)    
        myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1,ftol=1E-3,xtol=1E-6)    
        print("Status: ",myresults.status)
        mycost[myrow]=myresults.cost
        if xx.size > 5000:
            xxnew = np.arange(np.min(xx[mysamp]),np.max(xx[mysamp]),100)
            mypred = fitfun_sumofsin(xxnew,myresults.x)
            plt.plot(xxnew,mypred)
        else:
            mypred = fitfun_sumofsin(xx[mysamp],myresults.x)
            plt.plot(xx[mysamp],mypred)
    
    fig = plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(np.arange(1,myorder+1),mycost)
    #plt.xlim(gdata[0,0],gdata[-1,0])
    #if ylim is None:
    #    plt.ylim(0,20)
        
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight', dpi=200)
    
    fidx = mycost.argmin()+1
    
    lbb = np.squeeze(np.matlib.repmat(lb,1,fidx))
    ubb = np.squeeze(np.matlib.repmat(ub,1,fidx))
    p0 = np.divide(lbb+ubb,2)
        
    #scoef = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5) 
    scoef = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1,ftol=1E-3,xtol=1E-6)
    print("Sum of Sines Order Selected: ",fidx)
    
    return scoef.x, fidx


def fitfun_sumofsin_2angle(xxn,xxb,p): 
    myval=0
    for bb in np.arange(0,p.size-1,6):
        myval=myval + p[bb]*np.sin(p[bb+1]*xxn+p[bb+2]) + p[bb+3]*np.sin(p[bb+4]*xxb+p[bb+5])
    return myval
 
def function_sum_of_sin_2angle(xxn,xxb,yy,lb,ub,pp,ylim=None):

    def errfun(p, xxn,xxb, yy): return fitfun_sumofsin_2angle(xxn,xxb,p) - yy

    mykeep=np.isfinite(yy) & (np.abs(yy) < np.nanstd(yy)*3) & ~np.isnan(xxn) & ~np.isnan(xxb) & ~np.isnan(yy)
    xxn=xxn[mykeep]
    xxb=xxb[mykeep]
    yy=yy[mykeep]
    if xxn.size > 100000:
        mysamp = np.random.randint(0, xxn.size, 100000)
    else:
        mysamp = np.arange(0, xxn.size)
    
    fig = plt.figure(figsize=(7, 5), dpi=200)
    #fig.suptitle(title, fontsize=14)
    plt.plot(xxn[mysamp], yy[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')

    
    myorder=6
    mycost=np.zeros(myorder)
    for order in np.arange(1,myorder+1):
        print(order)
        myrow=order-1
        #lb = np.asarray([3, np.divide(2*np.pi,60000), -30])
        #ub = [30, np.divide(2*np.pi,6000), 30]
        lbb = np.squeeze(np.matlib.repmat(lb,1,order))
        ubb = np.squeeze(np.matlib.repmat(ub,1,order))
        p0 = np.divide(lbb+ubb,2)

        #p1, success, _ = optimize.least_squares(errfun, p0[:], args=([xdata], [ydata]), method='trf', bounds=([lb],[ub]), loss='soft_l1', f_scale=0.1)
        #myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5)    
        myresults = optimize.least_squares(errfun, p0, args=(xxn[mysamp], xxb[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='huber', f_scale=0.1,ftol=1E-5,xtol=1E-8)    
        print("Status: ",myresults.status)
        mycost[myrow]=myresults.cost

        xxn2 = np.linspace(np.min(xxn[mysamp]),np.max(xxn[mysamp]),100)
        xxb2 = np.linspace(np.min(xxb[mysamp]),np.max(xxb[mysamp]),100)
        mypred = fitfun_sumofsin_2angle(xxn2,xxb2,myresults.x)
        plt.plot(xxn2,mypred)
    
    fig = plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(np.arange(1,myorder+1),mycost)
    #plt.xlim(gdata[0,0],gdata[-1,0])
    #if ylim is None:
    #    plt.ylim(0,20)
        
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight', dpi=200)
    
    fidx = mycost.argmin()+1
    
    lbb = np.squeeze(np.matlib.repmat(lb,1,fidx))
    ubb = np.squeeze(np.matlib.repmat(ub,1,fidx))
    p0 = np.divide(lbb+ubb,2)
        
    #scoef = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5) 
    scoef = optimize.least_squares(errfun, p0, args=(xxn[mysamp],xxb[mysamp], yy[mysamp]), method='trf', bounds=([lbb,ubb]), loss='huber', f_scale=0.1,ftol=1E-5,xtol=1E-8)
    print("Sum of Sines Order Selected: ",fidx)
    
    return scoef.x, fidx
    



def plot_bias(data,gdata,title,pp,pmod=None,smod=None,plotmin=None):
    '''
    data : original data as numpy array (:,2), x=1col,y=2col
    gdata : grouped data as numpy array (:,2)
    pmod,smod are two model options to plot, numpy array (:,1)
    '''
    if data[:,0].size > 50000:
        mysamp = np.random.randint(0, data[:,0].size, 50000)
    else:
        mysamp = np.arange(0, data[:,0].size)
    mysamp=mysamp.astype(np.int64) #huh? 
    
    #title='Cross'
    fig = plt.figure(figsize=(7, 5), dpi=200)
    fig.suptitle(title, fontsize=14)
    if plotmin is None:
        plt.plot(data[mysamp,0], data[mysamp,1], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')
        plt.plot(gdata[:,0], gdata[:,1], '-', ms=2, color='0.15')
    else:
        plt.plot(gdata[:,0], gdata[:,1], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')
    plt.plot(gdata[:,0], np.zeros(gdata[:,0].size), 'k', ms=3)
    #plt.plot(gdata[:,0], gdata[:,1], '-', ms=2, color='0.15')

    if pmod is not None: 
        plt.plot(gdata[:,0],pmod, 'r-', ms=2)
    if smod is not None: 
        plt.plot(gdata[:,0],smod, 'm-', ms=2)
        
    plt.xlim(gdata[0,0],gdata[-1,0])
    #plt.ylim(-200,200)
    ymin, ymax = plt.ylim((np.nanmean(data[mysamp,1]))-2*np.nanstd(data[mysamp,1]),
                          (np.nanmean(data[mysamp,1]))+2*np.nanstd(data[mysamp,1]))
    # plt.axis([0, 360, -200, 200])
    plt.xlabel(title + 'track distance [meters]')
    plt.ylabel('dH [meters]')
    
    plt.show()
    pp.savefig(fig, bbox_inches='tight', dpi=200)
    pass

def correct_cross_track_bias(mst_dem,slv_dem,inang,pp):
    
    mytype = 'Cross'
    mytitle = 'dH Pre ' + mytype + '-track corrections'
    dH = calculate_dH(mst_dem,slv_dem)
    false_hillshade(dH,mytitle,pp)
    # #
    # Need conditional to check for large enough sample size... ?
    # #
    
    # calculate along/across track coordinates
    #myang = np.deg2rad(np.multiply(inang,np.multiply(dH,0)+1))# generate synthetic angle image for testing
    myang=np.deg2rad(inang.img)
    xxr, yyr = get_xy_rot(slv_dem,myang) #across,along track coordinates calculated from angle map

    # define original data with shape (:,2)
    orig_data = np.concatenate((xxr.reshape((xxr.size,1)),dH.img.reshape((dH.img.size,1))),1)
    
    # get group statistics of dH, and create matrix with same shape as orig_data
    xxgrp = get_group_statistics(orig_data[:,0],orig_data[:,1])
    grp_data = np.concatenate((xxgrp.index.values.reshape((xxgrp.index.size,1)),xxgrp.values[:,1].reshape((xxgrp.values[:,1].size,1))),1)
    
    # POLYNOMIAL FITTING
    pcoef, _ = polynomial_fit(grp_data[:,0],grp_data[:,1]) #mean
    #pcoef, _ = polynomial_fit(xxgrp.index.values,xxgrp.values[:,5]) #median
    # For testing the fitting using all values
    # pcoef2, _ = polynomial_fit(xxr[mykeep],dH[mykeep])
    polymod=poly.polyval(grp_data[:,0],pcoef)
    polyres=RMSE(grp_data[:,1]-polymod)
    print("Cross track Polynomial RMSE: ", polyres)
    
    plot_bias(orig_data,grp_data,mytype,pp,polymod)
    
    # Generate correction for DEM
    out_corr = poly.polyval(xxr,pcoef)
    
    # Correct DEM
    zupdate = np.ma.array(slv_dem.img + out_corr, mask=mst_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)
 
    dH1 = calculate_dH(mst_dem,slv_dem)
    final_histogram(dH.img,dH1.img,pp)

    
    return slv_dem, out_corr


def correct_along_track_bias(mst_dem,slv_dem,inang,pp):
    mytype = 'Along'
    mytitle = 'Pre ' + mytype + '-track corrections'
    dH = calculate_dH(mst_dem,slv_dem)
    false_hillshade(dH,mytitle,pp)
    # #
    # Need conditional to check for enough sample size... 
    # #
    
    # calculate along/across track coordinates
    #myang = np.deg2rad(np.multiply(inang,np.multiply(dH,0)+1))# generate synthetic angle image for testing
    myang=np.deg2rad(inang.img)
    xxr, yyr = get_xy_rot(mst_dem,myang) #across,along track coordinates calculated from angle map

    # define original data with shape (:,2)
    orig_data = np.concatenate((yyr.reshape((yyr.size,1)),dH.img.reshape((dH.img.size,1))),1)
    
    # get group statistics of dH, and create matrix with same shape as orig_data
    yygrp = get_group_statistics(orig_data[:,0],orig_data[:,1])
    grp_data = np.concatenate((yygrp.index.values.reshape((yygrp.index.size,1)),yygrp.values[:,1].reshape((yygrp.values[:,1].size,1))),1)
    
    # POLYNOMIAL FITTING
    pcoef, _ = polynomial_fit(grp_data[:,0],grp_data[:,1]) #mean
    polymod=poly.polyval(grp_data[:,0],pcoef)
    polyres=RMSE(grp_data[:,1]-polymod) 
    print("Along track Polynomial RMSE: ", polyres)
    
    # SUM OF SINES
    # First define the bounds of the three sine wave coefficients to solve
    lb = np.asarray([3, np.divide(2*np.pi,80000), -np.pi])
    ub = [20, np.divide(2*np.pi,20000), np.pi]
    scoef, _ = function_sum_of_sin(grp_data[:,0],grp_data[:,1],lb,ub,pp)
    sinmod = fitfun_sumofsin(grp_data[:,0],scoef) 
    #embed()
    sinres = RMSE(grp_data[:,1]-sinmod)
    print("Along track Sum_of_Sin RMSE: ", sinres)
    
    plot_bias(orig_data,grp_data,mytype,pp,polymod,sinmod)
    
    # ADD CONDITIONAL FOR CHOOSING WHICH FIT
    out_corr = poly.polyval(yyr,pcoef)
    out_corr2 = fitfun_sumofsin(yyr,scoef)
    
    res1 = RMSE(dH.img-out_corr)
    res2 = RMSE(dH.img-out_corr2)
    print("ALL Pixels Polynomial RMSE:", res1)
    print("ALL Pixels Sum_of_Sin RMSE:", res2)
    
    if sinres<=polyres:
        mycorr=out_corr2
    elif polyres<sinres:
        mycorr=out_corr
    
    zupdate = np.ma.array(slv_dem.img + mycorr, mask=mst_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)
    
    dH1 = calculate_dH(mst_dem,slv_dem)
    final_histogram(dH.img,dH1.img,pp)
    
    return slv_dem, mycorr


def correct_along_track_jitter(mst_dem,slv_dem,inang,pp):
    mytype = 'Along'
    mytitle = 'Pre ' + mytype + '-track jitter corrections'
    dH = calculate_dH(mst_dem,slv_dem)
    false_hillshade(dH,mytitle,pp)
    # #
    # Need conditional to check for enough sample size... 
    # #
    
    # calculate along/across track coordinates
    #myang = np.deg2rad(np.multiply(inang,np.multiply(dH,0)+1))# generate synthetic angle image for testing
    myang=np.deg2rad(inang.img)
    xxr, yyr = get_xy_rot(mst_dem,myang) #across,along track coordinates calculated from angle map

    # define original data with shape (:,2)
    orig_data = np.concatenate((yyr.reshape((yyr.size,1)),dH.img.reshape((dH.img.size,1))),1)
    
    # get group statistics of dH, and create matrix with same shape as orig_data
    yygrp = get_group_statistics(orig_data[:,0],orig_data[:,1],indist=200)
    grp_data = np.concatenate((yygrp.index.values.reshape((yygrp.index.size,1)),yygrp.values[:,5].reshape((yygrp.values[:,1].size,1))),1)
  
    # SUM OF SINES
    # First define the bounds of the three sine wave coefficients to solve
    lb = np.asarray([1, np.divide(2*np.pi,4800), -np.pi])
    ub = [3.5, np.divide(2*np.pi,3800), np.pi]
    scoef, _ = function_sum_of_sin(grp_data[:,0],grp_data[:,1],lb,ub,pp,ylim=300)
    sinmod = fitfun_sumofsin(grp_data[:,0],scoef) 
    #embed()
    sinres = RMSE(grp_data[:,1]-sinmod)
    print("Along track Sum_of_Sin RMSE: ", sinres)
    
    #plot_bias(orig_data,grp_data,mytype,pp)
    plot_bias(orig_data,grp_data,mytype,pp,None,sinmod,plotmin=1)

    out_corr = fitfun_sumofsin(yyr,scoef)
    
    res1 = RMSE(dH.img-out_corr)
    print("ALL Pixels Sum_of_Sin RMSE:", res1)
    
    zupdate = np.ma.array(slv_dem.img + out_corr, mask=mst_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)

    dH1 = calculate_dH(mst_dem,slv_dem)
    final_histogram(dH.img,dH1.img,pp)
    
    return slv_dem, out_corr

def correct_along_track_jitter2(mst_dem,slv_dem,inangN,inangB,pp):
    mytype = 'Along'
    mytitle = 'Pre ' + mytype + '-track jitter corrections'
    dH = calculate_dH(mst_dem,slv_dem)
    false_hillshade(dH,mytitle,pp)
    # #
    # Need conditional to check for enough sample size... 
    # #
    
    # calculate along/across track coordinates
    #myang = np.deg2rad(np.multiply(inang,np.multiply(dH,0)+1))# generate synthetic angle image for testing
    myangN=np.deg2rad(inangN.img)
    myangB=np.deg2rad(inangB.img)
    xxn, xxb = get_atrack_coord(mst_dem,myangN,myangB) #across,along track coordinates calculated from angle map

    # define original data with shape (:,2)
    orig_data = np.concatenate((xxn.reshape((xxn.size,1)),dH.img.reshape((dH.img.size,1))),1)
    
    # get group statistics of dH, and create matrix with same shape as orig_data
    #yygrp = get_group_statistics(orig_data[:,0],orig_data[:,1],indist=200)
    #grp_data = np.concatenate((yygrp.index.values.reshape((yygrp.index.size,1)),yygrp.values[:,5].reshape((yygrp.values[:,1].size,1))),1)
  
    # SUM OF SINES
    # First define the bounds of the three sine wave coefficients to solve
    lb = np.asarray([1, np.divide(2*np.pi,4800), -np.pi, 1, np.divide(2*np.pi,4800), -np.pi])
    ub = [3.5, np.divide(2*np.pi,3800), np.pi, 3.5, np.divide(2*np.pi,3800), np.pi]
    
    
    myslope = get_slope(slv_dem)
    dH.img[myslope.img>25]=np.nan
    
    xxn_vec = np.reshape(xxn,(xxn.size,1))
    xxb_vec = np.reshape(xxb,(xxb.size,1))
    dH_vec = np.reshape(dH.img,(dH.img.size,1))
    
    scoef, _ = function_sum_of_sin_2angle(xxn_vec,xxb_vec,dH_vec,lb,ub,pp,ylim=300)
    print(scoef)
    sinmod = fitfun_sumofsin_2angle(xxn_vec,xxb_vec,scoef) 
    #embed()
    sinres = RMSE(dH_vec-sinmod)
    print("Along track Sum_of_Sin RMSE: ", sinres)
    
    #plot_bias(orig_data,grp_data,mytype,pp)
    #plot_bias(orig_data,grp_data,mytype,pp,None,sinmod,plotmin=1)

    out_corr=np.reshape(sinmod,slv_dem.img.shape)
    fig = plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    #fig.suptitle(title, fontsize=14)
    plt1=plt.imshow(out_corr)
    plt1.set_clim(np.nanmin(out_corr),np.nanmax(out_corr))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plt1, cax=cax)
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight', dpi=200)
    #out_corr = fitfun_sumofsin(yyr,scoef)
    
    #res1 = RMSE(orig_data[:,1]-out_corr)
    #print("ALL Pixels Sum_of_Sin RMSE:", res1)
    
    zupdate = np.ma.array(slv_dem.img + out_corr, mask=mst_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)

    dH1 = calculate_dH(mst_dem,slv_dem)
    final_histogram(dH.img,dH1.img,pp)
    
    return slv_dem, out_corr

# the big kahuna
def mmaster_bias_removal(mst_dem,slv_dem,glac_mask=None,land_mask=None,cwd='.'):
    
    
    # import angle data
    #ang_map = GeoImg('TrackAngleMap_gt.tif')
    ang_mapN = GeoImg('TrackAngleMapN.tif')
    ang_mapB = GeoImg('TrackAngleMapB.tif')
    zupdate = np.ma.array(np.divide(ang_mapN.img + ang_mapB.img, 2))  # shift in z
    ang_mapNB = ang_mapN.copy(new_raster=zupdate)
      
    # pre-processing steps
    mst_coreg, slv_coreg, shift_params = preprocess(mst_dem, slv_dem)
    
    # create the output pdf
    pp = PdfPages('BiasCorrections_Results.pdf')
    
    # cross-track bias removal 
    # - assumes both dems include only stabile terrain. 
    # - Errors permitted as we will filter along the way
    slv_coreg_xcorr, xcorr = correct_cross_track_bias(mst_coreg,slv_coreg,ang_mapN,pp)
    plt.close("all")

    # along-track bias removal
    slv_coreg_xcorr_acorr, acorr = correct_along_track_bias(mst_coreg,slv_coreg_xcorr,ang_mapNB,pp)
    plt.close("all")

    # along-track jitter removal
    #slv_coreg_xcorr_acorr_jcorr, jcorr = correct_along_track_jitter(mst_coreg,slv_coreg_xcorr_acorr,ang_mapNB,pp)
    slv_coreg_xcorr_acorr_jcorr, jcorr = correct_along_track_jitter2(mst_coreg,slv_coreg_xcorr_acorr,ang_mapN,ang_mapB,pp)
    plt.close("all")


    # Calculate initial differences
    mytitle = 'dH Initial'
    dH0 = calculate_dH(mst_coreg,slv_coreg)
    false_hillshade(dH0,mytitle,pp)
    
    # Calculate post correction differences
    mytitle = 'dH Post corrections'
    dH_final = calculate_dH(mst_coreg,slv_coreg_xcorr_acorr_jcorr)
    false_hillshade(dH_final,mytitle,pp)
    
    dH0 = calculate_dH(mst_coreg,slv_coreg)
    final_histogram(dH0.img,dH_final.img,pp)

    # re-coregister
    recoreg_outdir = os.path.sep.join([cwd,'re-coreg'])
    mst_coreg, slv_adj_coreg, shift_params2 = dem_coregistration(mst_coreg, slv_coreg_xcorr_acorr_jcorr, glaciermask=glac_mask, 
                                                            landmask=land_mask, outdir=recoreg_outdir)
    plt.close("all")
    
    
    
    # clean-up 
    pp.close()
    print("Fin.")
    return mst_coreg, slv_coreg