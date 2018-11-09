from __future__ import print_function
from future_builtins import zip
from functools import partial
import os
import shutil
from glob import glob
import errno
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
from pybob.coreg_tools import dem_coregistration, false_hillshade, get_slope, final_histogram, create_stable_mask
from pybob.GeoImg import GeoImg
from pybob.image_tools import nanmedian_filter
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
        feature.SetGeometry(ogr.CreateGeometryFromWkt(inpoly.wkt))
#        feature.SetGeometry(ogr.CreateGeometryFromWkt(inpoly.crs_wkt))
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


def get_group_statistics(invar, indata, indist=500):
    """
    Calculate statistics on groups of pixels.
    """
    xxid = make_group_id(invar, indist)  # across track coordinates (grouped 500m)
    # yyid = make_group_id(yyr,500) # along track coordinates (grouped 500m)

    # Calculate group statistics 
    mykeep = np.isfinite(indata)
    data = pd.DataFrame({'dH': indata[mykeep], 'XX': xxid[mykeep]})
    xxgrp = data['dH'].groupby(data['XX']).describe()

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
    
    ### MADE AN ADJUSTMENT TO ASSUME THAT THE .MET FILE IS IN THE CURRENT FOLDER OF OPERATION
    metlist = glob(os.path.abspath('*.met'))

    schema = {'properties': [('id', 'int')], 'geometry': 'Polygon'}
    outshape = fiona.open(gran_name + '_Footprint.shp', 'w', crs=fiona.crs.from_epsg(int(epsg)), driver='ESRI Shapefile', schema=schema)

    footprints = []
    for m in metlist:
        clean = [line.strip() for line in open(m).read().split('\n')]

        if os.path.sep in m:
            m = m.split(os.path.sep)[-1]

        latinds = [i for i, line in enumerate(clean) if 'GRingPointLatitude' in line]
        loninds = [i for i, line in enumerate(clean) if 'GRingPointLongitude' in line]

        latlines = clean[latinds[0]:latinds[1] + 1]
        lonlines = clean[loninds[0]:loninds[1] + 1]

        lonvalstr = lonlines[2]
        latvalstr = latlines[2]

        lats = [float(val) for val in latvalstr.strip('VALUE  = ()').split(',')]
        lons = [float(val) for val in lonvalstr.strip('VALUE  = ()').split(',')]

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
    x = x[:-1]  # drop the last coordinate, which is a duplicate of the first
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

    lside = range(len(x) - 1, lower_left - 1, -1)
    rside = range(upper_right, lower_right + 1)

    left_side = LineString(list(zip(np.array(x)[lside], np.array(y)[lside])))
    right_side = LineString(list(zip(np.array(x)[rside], np.array(y)[rside])))

    lproj = left_side.interpolate(track_dist)
    rproj = right_side.interpolate(track_dist)
    # get the angle of the line formed by connecting lproj, rproj
    dx = lproj.x - rproj.x
    dy = lproj.y - rproj.y

    return 90 + np.rad2deg(np.arctan(dx / dy))


def preprocess(mst_dem, slv_dem, glacmask=None, landmask=None, cwd='.',out_dir='.', pts=False):
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
    glacmask : string, optional
        Path to shapefile representing points to exclude from co-registration
        consideration (i.e., glaciers).
    landmask : string, optional
        Path to shapefile representing points to include in co-registration
        consideration (i.e., stable ground/land).
    cwd : string, optional
        Working directory to use [Assumes '.'].
    out_dir : string, optional
        Output directory for the new coregistration folder    
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
    ast_name = slv_dem.rsplit('_Z.tif')[0]
    mask_name = os.path.sep.join([cwd, '{}_CORR.tif'.format(ast_name)])
    slv_name = os.path.sep.join([cwd, '{}_filtZ.tif'.format(ast_name)])
    mask_raster_threshold(slv_dem, mask_name, slv_name, 60, np.float32)

    # co-register master, slave
    out_dir2 = os.path.sep.join([cwd, 'coreg'])
    mst_coreg, slv_coreg, shift_params = dem_coregistration(mst_dem, slv_name, glaciermask=glacmask,
                                                            landmask=landmask, outdir=out_dir2, pts=pts)
#    print(slv_coreg.filename)
    # remove coreg folder, save slv as *Zadj1.tif, but save output.pdf
#    shutil.move(os.path.sep.join([out_dir, 'CoRegistration_Results.pdf']),
#                os.path.sep.join([cwd, 'CoRegistration_Results.pdf']))
#    shutil.move(os.path.sep.join([out_dir, 'coreg_params.txt']), os.path.sep.join([cwd, 'coreg_params.txt']))
#    shutil.rmtree(out_dir)
    
    ### EXPORT and write files. Move directory to main Bias Corrections directory
    ### *** NOTE: GeoImg.write replaces the mask !!!***
    slv_coreg.write(os.path.sep.join([out_dir2, '{}_Z_adj.tif'.format(ast_name)]))
    # shift ortho, corr masks, save as *adj1.tif
    ortho = GeoImg(ast_name + '_V123.tif')
    corr = GeoImg(ast_name + '_CORR.tif')

    ortho.shift(shift_params[0], shift_params[1])
    ortho.write(os.path.sep.join([out_dir2, '{}_V123_adj.tif'.format(ast_name)]))

    corr.shift(shift_params[0], shift_params[1])
    corr.write(os.path.sep.join([out_dir2, '{}_CORR_adj.tif'.format(ast_name)]), dtype=np.uint8)

    # Move coreg folder to output directory
    shutil.move(out_dir2, out_dir)
    plt.close("all")

    return mst_coreg, slv_coreg, shift_params


def mask_raster_threshold(rasname, maskname, outfilename, threshold=50, datatype=np.float32):
    """
    filters values in rasname if maskname is less then threshold
    """
    myras = GeoImg(rasname)
    mymask = GeoImg(maskname)

    rem = mymask.img < threshold
    myras.img[rem] = np.nan
    myras.write(outfilename, driver='GTiff', dtype=datatype)

    return myras


def RMSE(indata):
    """ Return root mean square of indata."""
    
    # add 3x std dev filter
    indata[np.abs(indata)>3*np.nanstd(indata)] = np.nan
    
    myrmse = np.sqrt(np.nanmean(np.square(np.asarray(indata))))
    return myrmse


def calculate_dH(mst_dem, slv_dem, pts):
    # mst_dem and slv_dem must be GeoIMG objects with same pixels
    # 1) calculates the differences
    # 2) Runs a 5x5 pixel median filter
    # 3) Hard Removes values > 100

    if not pts:
        
        zupdate = np.ma.array(mst_dem.img.data - slv_dem.img.data, mask=slv_dem.img.mask)
#        zupdate2 = np.ma.array(ndimage.median_filter(zupdate, 7), mask=slv_dem.img.mask)
        zupdate2 = np.ma.array(nanmedian_filter(zupdate, size=7), mask=slv_dem.img.mask)
        dH = slv_dem.copy(new_raster=zupdate2)
#        dH.mask(slv_dem.img.mask)
        
        master_mask = isinstance(mst_dem.img, np.ma.masked_array)
        slave_mask = isinstance(slv_dem.img, np.ma.masked_array)

        myslope = get_slope(slv_dem)
        fmask = np.logical_or.reduce((np.greater(np.abs(dH.img), 100), np.less(myslope.img,0.5), np.greater(myslope.img, 25)))

        dH.mask(fmask)
#        dH.mask(np.abs(dH.img) > 100)   
#        dH2.mask(np.logical_or(myslope.img < 0.5, myslope.img > 25))

        if master_mask and slave_mask:
            dH.mask(np.logical_or(mst_dem.img.mask, slv_dem.img.mask))
        elif master_mask:
            dH.mask(mst_dem.img.mask)
        elif slave_mask:
            dH.mask(slv_dem.img.mask)
        
#        dH2.mask(dH.img.mask)           

 
        
#        zupdate = np.ma.array(mst_dem.img.data - slv_dem.img, mask=slv_dem.img.mask)
##        dH = slv_dem.copy(new_raster=zupdate)
#        zupdate2 = np.ma.array(ndimage.median_filter(zupdate, 7), mask=slv_dem.img.mask)
#        dH2 = slv_dem.copy(new_raster=zupdate2)
#        # dH = master.copy(new_raster = (master.img-slave.img))
        
#        mykeep = ((np.absolute(dH.img) < 200.0) & np.isfinite(dH.img) &
#                  (slope > 7.0) & (dH.img != 0.0) & (aspect >= 0))
#        dH.img[np.invert(mykeep)] = np.nan

#        plt.figure(figsize=(5, 5))
#        plt.imshow(dH.img, interpolation='nearest', cmap='gray')
#        plt.show()
    elif pts:
        # NEED TO CHECK THE MASKING
        slave_pts = slv_dem.raster_points(mst_dem.xy, mode='quintic')
        dH = mst_dem.elev - slave_pts

        myslope = get_slope(slv_dem)
        slope_pts = myslope.raster_points(mst_dem.xy, mode='cubic')
        
#        fmask = np.logical_or((np.isnan(dH2.img), np.abs(dH2.img[fmask]) > 40))
        fmask = np.logical_or.reduce((np.greater(np.abs(dH), 100), np.less(slope_pts,0.5), np.greater(slope_pts, 25)))

        dH[fmask] = np.nan

    return dH


def get_xy_rot(mst_dem, myang):
    # creates matrices for along and across track distances from a reference dem and a raster angle map (in radians)

    xx, yy = mst_dem.xy(grid=True)
    xx = xx - np.min(xx)
    yy = yy - np.min(yy)
    # xxr = np.multiply(xx,np.rad2deg(np.cos(myang))) + np.multiply(-1*yy,np.rad2deg(np.sin(myang)))
    # yyr = np.multiply(xx,np.rad2deg(np.sin(myang))) + np.multiply(yy,np.rad2deg(np.cos(myang)))
    xxr = np.multiply(xx, np.cos(myang)) + np.multiply(-1 * yy, np.sin(myang))
    yyr = np.multiply(xx, np.sin(myang)) + np.multiply(yy, np.cos(myang))

    # TO USE FOR INITIALIZING START AT ZERO
    xxr = xxr - np.nanmin(xxr)
    yyr = yyr - np.nanmin(yyr)

    plt.figure(figsize=(5, 5))
    plt.imshow(xxr, interpolation='nearest')
    # plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(yyr, interpolation='nearest')
    # plt.show()

    return xxr, yyr


def get_atrack_coord(mst_dem, myangN, myangB):
    # creates matrices for along and across track distances from a reference dem and a raster angle map (in radians)

    xx, yy = mst_dem.xy(grid=True)
    xx = xx - np.min(xx)
    yy = yy - np.min(yy)
    # xxr = np.multiply(xx,np.rad2deg(np.cos(myang))) + np.multiply(-1*yy,np.rad2deg(np.sin(myang)))
    # yyr = np.multiply(xx,np.rad2deg(np.sin(myang))) + np.multiply(yy,np.rad2deg(np.cos(myang)))
    # xxr = np.multiply(xx,np.cos(myang)) + np.multiply(-1*yy,np.sin(myang))

    yyn = np.multiply(xx, np.sin(myangN)) + np.multiply(yy, np.cos(myangN))
    yyb = np.multiply(xx, np.sin(myangB)) + np.multiply(yy, np.cos(myangB))

    plt.figure(figsize=(5, 5))
    plt.imshow(yyn, interpolation='nearest')
    # plt.show()

    plt.figure(figsize=(5, 5))
    plt.imshow(yyb, interpolation='nearest')
    # plt.show()

    return yyn, yyb


def get_fit_variables(mst_dem, slv_dem, xxn, pts, pp, xxb=None, mytype='Unknown'):
    mytitle = 'dH Pre ' + mytype + '-track corrections'

    if xxb is None:
       
        if not pts:
            dHmat = calculate_dH(mst_dem, slv_dem, pts)
            false_hillshade(dHmat, mytitle, pp)

            xx = xxn.reshape((1, xxn.size))
            dH = dHmat.img.reshape((1, dHmat.img.size))

            mynan = np.logical_and.reduce((np.invert(np.isfinite(dH)), np.invert(np.isfinite(xx)), (np.abs(dH) > np.nanstd(dH) * 3)))
#            mynan = (np.abs(dH) > np.nanstd(dH) * 3)
#            dH[mynan] = np.nan
            dH = dH[~mynan]
            xx = xx[~mynan]
            
            # get group statistics of dH, and create matrix with same shape as orig_data
            grp_sts = get_group_statistics(xx, dH, indist=500)
            grp_xx = grp_sts.index.values
            grp_dH = grp_sts.values[:, 1]
        elif pts:
            XXR = slv_dem.copy(new_raster=xxn)
            xx = XXR.raster_points(mst_dem.xy, mode='cubic')

            dH = calculate_dH(mst_dem, slv_dem, pts)

            # Add light filtering (remove outliers, and start/end of xx for edge effects)
#            xxlim = 2000
#            mykeep = (abs(dH) < np.nanstd(dH) * 3) & (xx > xxlim) & (xx < np.nanmax(xx) - xxlim)
            mynan = np.logical_and.reduce((np.invert(np.isfinite(dH)), np.invert(np.isfinite(xx)), (np.abs(dH) > np.nanstd(dH) * 3)))
            xx[mynan] = np.nan
            dH[mynan] = np.nan

            # get group statistics of dH, and create matrix with same shape as orig_data
            grp_sts = get_group_statistics(xx, dH, indist=500)
            grp_xx = grp_sts.index.values
            grp_dH = grp_sts.values[:, 1]
        return np.squeeze(xx), np.squeeze(dH), grp_xx, grp_dH

    elif xxb is not None:
 
        if not pts:
            dHmat = calculate_dH(mst_dem, slv_dem, pts)
            false_hillshade(dHmat, mytitle, pp)
            xx1 = xxn.reshape((1, xxn.size))
            xx2 = xxb.reshape((1, xxb.size))
            dH = dHmat.img.reshape((1, dHmat.img.size))

#            mynan = (np.abs(dH) > np.nanstd(dH) * 3)
            mynan = np.logical_and.reduce((np.invert(np.isfinite(dH)), np.invert(np.isfinite(xx1)), np.invert(np.isfinite(xx2)), (np.abs(dH) > np.nanstd(dH) * 3)))
            dH[mynan] = np.nan

            # get group statistics of dH, and create matrix with same shape as orig_data
            grp_sts = get_group_statistics(xx1, dH, indist=100)
            grp_xx = grp_sts.index.values
            grp_dH = grp_sts.values[:, 1]

        elif pts:
            XXR = slv_dem.copy(new_raster=xxn)
            XXR2 = slv_dem.copy(new_raster=xxb)

            xx1 = XXR.raster_points(mst_dem.xy, mode='cubic')
            xx2 = XXR2.raster_points(mst_dem.xy, mode='cubic')
            dH = calculate_dH(mst_dem, slv_dem, pts)
            
            mynan = np.logical_and.reduce((np.invert(np.isfinite(dH)), np.invert(np.isfinite(xx1)), np.invert(np.isfinite(xx2)), (np.abs(dH) > np.nanstd(dH) * 3)))
            dH[mynan] = np.nan

            
            # get group statistics of dH, and create matrix with same shape as orig_data
            grp_sts = get_group_statistics(xx1, dH, indist=500)
            grp_xx = grp_sts.index.values
            grp_dH = grp_sts.values[:, 1]

        return xx1, xx2, dH, grp_xx, grp_dH


def fitfun_polynomial(xx, params):
    #    myval=0
    #    for i in np.arange(0,params.size):
    #        myval = myval + params[i]*(xx**i)
    # myval=myval + params[i]*(xx**i)
    return sum([p * (np.divide(xx,1000) ** i) for i, p in enumerate(params)])
#    return sum([p * (xx ** i) for i, p in enumerate(params)])

def robust_polynomial_fit(xx, yy):

    print("Original Sample size :", yy.size)
    # mykeep=np.isfinite(yy) and np.isfinite(xx)
    mykeep = np.logical_and(np.isfinite(yy), np.isfinite(xx))
#    mykeep = np.logical_and.reduce((np.isfinite(yy), np.isfinite(xx), (np.abs(yy) < np.nanstd(yy) * 3))) 
    xx = xx[mykeep]
    yy = yy[mykeep]
    
    print("Final Sample size :", yy.size)
    print("Remaining NaNs :", np.sum(np.isnan(yy)))
    sampsize = 50000 #np.int(np.floor(xx.size*0.25))
    if xx.size > sampsize:
        mysamp = np.random.randint(0, xx.size, sampsize)
    else:
        mysamp = np.arange(0, xx.size)

    fig = plt.figure(figsize=(7, 5), dpi=200)
    # fig.suptitle(title, fontsize = 14)
    plt.plot(xx[mysamp], yy[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')

    myorder = 4
    mycost = np.empty(myorder)
    coeffs = np.zeros((myorder, myorder + 1))
    xnew = np.arange(np.nanmin(xx), np.nanmax(xx), 1000)

    def errfun(p, xx, yy):
        return fitfun_polynomial(xx, p) - yy

    for deg in np.arange(1, myorder + 1):
#        p0 = np.zeros(deg + 1)
        p0, _ = poly.polyfit(np.divide(xx[mysamp],1000), yy[mysamp], deg, full=True)
#        print(p0)
#        lbb = np.zeros(deg + 1)-1000
#        ubb = np.zeros(deg + 1)+1000

#        myresults = optimize.leastsq(errfun, p0, args=(xx[mysamp], yy[mysamp]))
        myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', loss='soft_l1',
                                           f_scale=0.1, ftol=1E-8, xtol=1E-8)
#        print("Status: ", myresults.status)
        print("Polynomial degree - ",deg, " --> Status: ", myresults.success," - ",myresults.status)
        print(myresults.message)
        print("Lowest cost:", myresults.cost)
        print("Parameters:", myresults.x)

        mycost[deg - 1] = myresults.cost
        coeffs[deg - 1, 0:myresults.x.size] = myresults.x
        # if xx.size > 5000:
        mypred = fitfun_polynomial(xnew, myresults.x)
        plt.plot(xnew, mypred)

        # else:
        #    mypred = fitfun_sumofsin(xx[mysamp],myresults.x)
        #    plt.plot(xx[mysamp],mypred)
    fidx = mycost.argmin()
    plt.ylim(-75,75)
    
    # This is to check whether percent improvement is a better way to choose the best fit. 
    # For now, comment out... 
    #    perimp=np.divide(mycost[:-1]-mycost[1:],mycost[:-1])*100
    #    fmin=np.asarray(np.where(perimp>5))
    #    if fmin.size!=0:
    #        fidx = fmin[0,-1]+1
    # print('fidx: {}'.format(fidx))

    print("Polynomial Order Selected: ", fidx + 1)

    return np.trim_zeros(coeffs[fidx],'b'), fidx + 1


def polynomial_fit(x, y):
    # edge removal
#    x = x[5:-5]
#    y = y[5:-5]

    max_poly_order = 6

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, '.')
    rmse = np.empty(max_poly_order - 1)
    coeffs = np.zeros((max_poly_order - 1, max_poly_order + 1))
    xnew = np.arange(np.nanmin(x), np.nanmax(x), 1000)
    for deg in np.arange(2, max_poly_order + 1):
        # print(deg)
        p, r = poly.polyfit(x, y, deg, full=True)
        p2 = poly.polyval(xnew, p)
        plt.plot(xnew, p2)
        coeffs[deg - 2, 0:p.size] = p
        rmse[deg - 2] = np.sqrt(np.divide(r[0], y.size - deg))

    plt.figure(figsize=(7, 5))
    plt.plot(np.arange(2, max_poly_order + 1), rmse)

    # Choose order of polynomial - 2 options 
    # a) lowest RMSE or 
    # b) by checking the percent improvemnt (more robust?)

    # a) lowest RMSE
    fidx = rmse.argmin()

    # b) [DEFAULT??] find the highest order which gave a 5% improvemnt in the RMSE
    perimp = np.divide(rmse[:-1] - rmse[1:], rmse[:-1]) * 100
    # print(perimp)
    # print(rmse)
    fmin = np.asarray(np.where(perimp > 5))
    if fmin.size != 0:
        fidx = fmin[0, -1] + 1
    # print('fidx: {}'.format(fidx))

    print("Polynomial Order Selected: ", fidx + 2)
    return coeffs[fidx], rmse[fidx]

class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
#        while True:
            # this could be done in a much more clever way, but it will work for example purposes
#            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
#            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin):
#                break
        return np.clip(x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x)), self.xmin, self.xmax)


def fitfun_sumofsin(xx, p):
#    myval = 0
#    for bb in np.arange(0, p.size - 1, 3):
#        myval = myval + p[bb] * np.sin(p[bb + 1] * xx + p[bb + 2])
#        
    p = np.asarray(p)
    aix = np.arange(0, p.size, 3)
    bix = np.arange(1, p.size, 3)
    cix = np.arange(2, p.size, 3)
#    if len(xx.shape) == 1:
#        myval = np.sum(p[aix] * np.sin(p[bix] * xx[:,np.newaxis] + p[cix]),axis=1)
#    elif len(xx.shape) == 2:
#        myval = np.sum(p[aix] * np.sin(p[bix] * xx[:,:,np.newaxis] + p[cix]),axis=2)
    if len(xx.shape) == 1:
        myval = np.sum(p[aix] * np.sin(np.divide(2*np.pi,p[bix]) * np.divide(xx[:,np.newaxis],1000) + p[cix]),axis=1)
    elif len(xx.shape) == 2:
        myval = np.sum(p[aix] * np.sin(np.divide(2*np.pi,p[bix]) * np.divide(xx[:,:,np.newaxis],1000) + p[cix]),axis=2)
       
        
        
    return myval

def fitfun_sumofsin2(p,xx,yy):

    myval = fitfun_sumofsin(xx, p)      
    # DEFINE THE COST FUNCTION        
#    myerr = RMSE(yy - myval)
#    myerr = nmad(yy - myval)
#    myerr = np.sum(np.abs(yy-myval))    # called MAE or L1 loss function
#    myerr = np.linalg.norm(yy-myval)
#    myerr = (np.sum((yy-myval) ** 2))
    myerr = np.sum(2 * (np.sqrt(1+np.square(yy-myval))-1))    # SOFT-L1 loss function (reduce the weight of outliers)
    
    return myerr


def function_sum_of_sin(xx, yy, lb, ub, pp, ylim=None):
    
    # Define the error function 
    #
    def errfun(p, xx, yy):
        return fitfun_sumofsin(xx, p) - yy

    mykeep = np.logical_and.reduce((np.isfinite(yy), np.isfinite(xx), (np.abs(yy) < np.nanstd(yy) * 3))) 
#    mykeep = np.logical_and(np.isfinite(yy), np.isfinite(xx))
    xx = xx[mykeep]
    yy = yy[mykeep]
    #sampsize = np.int(np.floor(xx.size*0.25)) # for use as a percentage
    sampsize = 15000
    if xx.size > sampsize:
        mysamp = np.random.randint(0, xx.size, sampsize)
    else:
        mysamp = np.arange(0, xx.size)
        
    fig = plt.figure(figsize=(7, 5), dpi=200)
    # fig.suptitle(title, fontsize=14)
    plt.plot(xx[mysamp], yy[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full',label="data")
    print("Sum of Sines Fitting using ", mysamp.size, "samples")
#    print("Initial MAE value:",np.sum(np.abs(yy[mysamp])))
    myorder = 10
    coeffs = np.zeros((myorder, myorder * 3))
    mycost = np.zeros(myorder)
    xxnew = np.arange(np.min(xx[mysamp]), np.max(xx[mysamp]), 100)
#    p_init = np.asarray([5,30,0])
    p_init = np.divide(lb + ub, 2)
    for order in np.arange(1, myorder + 1):
        myrow = order - 1
 
        # create the bounds matrices based upon the order of the fit. 
#        lbb = np.squeeze(np.matlib.repmat(lb, 1, order))
#        ubb = np.squeeze(np.matlib.repmat(ub, 1, order))
        lbb = np.tile(lb,order)
        ubb = np.tile(ub,order)
#        take_step = RandomDisplacementBounds(lbb,ubb)
        # If the first iteration, set initial parameters to the average of the bounds
        # otherwise, use the output from the previous sum of sines order
#        if order <= 1:
        p0 = np.tile(p_init,order)
        
#        p0 = np.divide(lbb + ubb, 2)
#        p0[0::3] = np.repeat(5,p0[0::3].size)
#        p0 = np.squeeze(p0) 
#        else:
#            p0 = np.ones(lbb.size)
#            p0[0:myresults.x.size] = myresults.x
#            p0[myresults.x.size:] = np.divide(lbb[myresults.x.size:] + ubb[myresults.x.size:], 2)
        # p1, success, _ = optimize.least_squares(errfun, p0[:], args=([xdata], [ydata]),
        #  method='trf', bounds=([lb],[ub]), loss='soft_l1', f_scale=0.1)
        # myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]),
        #  method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5)

        myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]), method='trf', bounds=([lbb, ubb]), 
                                           loss='soft_l1', f_scale=0.1, ftol=1E-8, xtol=1E-8)

#        bounds=((lbb[0],ubb[0]),(lbb[1],ubb[1]),(lbb[2],ubb[2]))
#        options={'maxiter': 1000,'maxfun': 500,'disp':True} 
#        options={'maxiter': 1000,'minfev':RMSE(yy)-5, 'eta':0.6}
#        myresults = optimize.minimize(fitfun_sumofsin2, p0, args=(xx[mysamp], yy[mysamp]),
#                                      bounds=optimize.Bounds(lbb,ubb), method='TNC',
#                                      options={'maxiter': 1000,'ftol':1E-1})
#        print("Sum of Sines Order ",order, " --> Status: ", myresults.success," - ",myresults.status)
#        print(myresults.message)
#        print(myresults)
        
##        if myresults.success == False:
#        minimizer_kwargs = dict(args=(xx[mysamp], yy[mysamp]), method="L-BFGS-B", bounds=optimize.Bounds(lbb,ubb), options={'ftol':1E-3} )
#        myresults = optimize.basinhopping(fitfun_sumofsin2, p0, T=100, niter=200, niter_success=75, minimizer_kwargs=minimizer_kwargs)
##        print(myresults)
#        myresults = myresults.lowest_optimization_result
        
        print("Sum of Sines Order ",order, " --> Status: ", myresults.success," - ",myresults.status)
        print(myresults.message)
        print("Lowest cost:", myresults.cost)
#        print("Lowest cost:", myresults.fun)
        print("Parameters:",myresults.x)
        
       # Get the cost function
#        mycost[myrow] = myresults.cost
        mycost[myrow] = fitfun_sumofsin2(myresults.x,xx[mysamp],yy[mysamp])

        coeffs[order - 1, 0:myresults.x.size] = myresults.x
        mypred = fitfun_sumofsin(xxnew, myresults.x)
        plt.plot(xxnew, mypred,label="order={0}".format(order))
    plt.legend(loc=0)
    


    fig = plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(np.arange(1, myorder + 1), mycost)
    # plt.xlim(gdata[0,0],gdata[-1,0])
    # if ylim is None:
    #    plt.ylim(0,20)

    plt.xlabel('Sum of Sines Order')
    plt.ylabel('Cost function')
#    plt.legend(('Raw [samples]', 'Grouped Median', 'Fit'), loc=1)
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight', dpi=200)

    fidx = mycost.argmin()
    print("Sum of Sines Order Selected: ", fidx + 1)

    #    lbb = np.squeeze(np.matlib.repmat(lb,1,fidx))
    #    ubb = np.squeeze(np.matlib.repmat(ub,1,fidx))
    #    p0 = np.divide(lbb+ubb,2)
    #
    #    scoef = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]),
    #        method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5)
    #    scoef = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]),
    #        method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1,ftol=1E-3,xtol=1E-6)

    return np.asarray(coeffs[fidx]), fidx + 1


def fitfun_sumofsin_2angle(xxn, xxb, p):
    p = np.squeeze(np.asarray(p))
    aix = np.arange(0, p.size, 6)
    bix = np.arange(1, p.size, 6)
    cix = np.arange(2, p.size, 6)
    dix = np.arange(3, p.size, 6)
    eix = np.arange(4, p.size, 6)
    fix = np.arange(5, p.size, 6)

#        myval = np.sum(p[aix] * np.sin(np.divide(2*np.pi,p[bix]) * np.divide(xx[:,np.newaxis],1000) + p[cix]),axis=1)

    if len(xxn.shape) == 1:
        myval = np.sum(p[aix] * np.sin(np.divide(2*np.pi,p[bix]) * np.divide(xxn[:,np.newaxis],1000) + p[cix]) + p[dix] * np.sin(np.divide(2*np.pi, p[eix]) * np.divide(xxb[:,np.newaxis],1000) + p[fix]),axis=1)
    elif len(xxn.shape) == 2:
        myval = np.sum(p[aix] * np.sin(np.divide(2*np.pi,p[bix]) * np.divide(xxn[:,:,np.newaxis],1000) + p[cix]) + p[dix] * np.sin(np.divide(2*np.pi, p[eix]) * np.divide(xxb[:,:,np.newaxis],1000) + p[fix]),axis=2)
        
    return myval

def huber_loss(z):
    out = np.asarray(np.square(z)*1.000)
    out[np.where(z>1)] = 2*np.sqrt(z[np.where(z>1)])- 1
    return out.sum()    


def fitfun_sumofsin_2angle2(p, xxn, xxb, yy):
    
    myval = fitfun_sumofsin_2angle(xxn, xxb, p)

   # DEFINE THE COST FUNCTION        
#    myerr = RMSE(yy - myval)
#    myerr = nmad(yy - myval)
#    myerr = np.sum(np.abs(yy-myval))    # called MAE or L1 loss function
#    myerr = np.linalg.norm(yy-myval)
#    myerr = np.sqrt(np.sum((yy-myval) ** 2))
#    myerr = huber_loss(yy-myval)    # HUBER loss function (reduce the weight of outliers)
    myerr = np.sum(2 * (np.sqrt(1+np.square(yy-myval))-1))    # SOFT-L1 loss function (reduce the weight of outliers)
    
    
    return myerr

def function_sum_of_sin_2angle(xxn, xxb, yy, lb, ub, pp, ylim=None):
    def errfun(p, xxn, xxb, yy):
        return fitfun_sumofsin_2angle(xxn, xxb, p) - yy

    print("Original Sample Size: ", xxn.size)
#    mykeep = np.isfinite(yy) & np.isfinite(xxn) & np.isfinite(xxb) & (np.abs(yy) < np.nanstd(yy) * 3)
#    mykeep = np.asarray(np.isfinite(yy) and np.isfinite(xxn) and np.isfinite(xxb) and (np.abs(yy) < np.nanstd(yy) * 3))
    mykeep = np.logical_and.reduce((np.isfinite(yy), np.isfinite(xxn), np.isfinite(xxb), (np.abs(yy) < np.nanstd(yy) * 3)))
    xxn = np.squeeze(xxn[mykeep])
    xxb = np.squeeze(xxb[mykeep])
    yy = np.squeeze(yy[mykeep])

    print("Filtered Sample Size: ", xxn.size)
    #sampsize = np.int(np.floor(xx.size*0.25)) # for use as a percentage
    sampsize = 100000
    if xxn.size > sampsize:
        mysamp = np.random.randint(0, xxn.size, sampsize)
    else:
        mysamp = np.arange(0, xxn.size)
    print("Sum of Sines Fitting using ", mysamp.size, "samples")

    fig = plt.figure(figsize=(7, 5), dpi=200)
    # fig.suptitle(title, fontsize = 14)
    plt.plot(xxn[mysamp], yy[mysamp], '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')

    myorder = 10
    mycost = np.zeros(myorder)
    coeffs = np.zeros((myorder, myorder * 6))
    if xxn.max() < 100000:
        p_init = np.asarray([1,5,0,1,5,0])
    else:
        p_init = np.asarray([3,30,0,3,30,0])
        
    print(p_init)        
    for order in np.arange(1, myorder + 1):
        myrow = order - 1
        # lb = np.asarray([3, np.divide(2*np.pi,60000), -30])
        # ub = [30, np.divide(2*np.pi,6000), 30]
#        lbb = np.squeeze(np.matlib.repmat(lb, 1, order))
#        ubb = np.squeeze(np.matlib.repmat(ub, 1, order))
        lbb = np.tile(lb,order)
        ubb = np.tile(ub,order)
#        take_step = RandomDisplacementBounds(lbb,ubb)
        # If the first iteration, set initial parameters to the average of the bounds
        # otherwise, use the output from the previous sum of sines order
#        if order <= 2:
#        p0 = np.divide(lbb + ubb, 2)
        p0 = np.tile(p_init,order)
#        else:
#            p0 = np.ones(lbb.size)
#            p0[0:myresults.x.size] = myresults.x
#            p0[myresults.x.size:] = np.divide(lbb[myresults.x.size:] + ubb[myresults.x.size:], 2)

        # p1, success, _ = optimize.least_squares(errfun, p0[:], args=([xdata], [ydata]),
        #  method='trf', bounds=([lb],[ub]), loss='soft_l1', f_scale=0.1)
        # myresults = optimize.least_squares(errfun, p0, args=(xx[mysamp], yy[mysamp]),
        # method='trf', bounds=([lbb,ubb]), loss='soft_l1', f_scale=1.5)
        
        myresults = optimize.least_squares(errfun, p0, args=(xxn[mysamp], xxb[mysamp], yy[mysamp]),
                                           method='trf', bounds=([lbb, ubb]), loss='soft_l1',
                                           f_scale=10, ftol=1E-8, xtol=1E-8, tr_solver='lsmr')
#        print("Sum of Sines Order ",order, " --> Status: ", myresults.status)
#        

#        myresults = optimize.minimize(fitfun_sumofsin_2angle2, p0, args=(xxn[mysamp], xxb[mysamp], yy[mysamp]),
#                                      bounds=optimize.Bounds(lbb,ubb), method='L-BFGS-B',
#                                      options={'maxiter': 1000,'maxfun':1000, 'ftol':1E-8})
#        print("Sum of Sines Order ",order, " --> Status: ", myresults.success," - ",myresults.status)
#        print(myresults.message)

#        myresults = optimize.minimize(fitfun_sumofsin_2angle2, p0, args=(xxn[mysamp], xxb[mysamp], yy[mysamp]),
#                                      bounds=optimize.Bounds(lbb,ubb), method='L-BFGS-B',
#                                      options={'maxiter': 2000, 'ftol':1E-4})

#        minimizer_kwargs = dict(args=(xxn[mysamp], xxb[mysamp], yy[mysamp]), method="L-BFGS-B", bounds=optimize.Bounds(lbb,ubb), options={"ftol":1E-4, "maxiter":1000})
#        myresults = optimize.basinhopping(fitfun_sumofsin_2angle2, p0, T=100, niter=200, niter_success=75, minimizer_kwargs=minimizer_kwargs)
##        minimizer_kwargs = dict(args=(xxn[mysamp], xxb[mysamp], yy[mysamp]), method="L-BFGS-B", bounds=optimize.Bounds(lbb,ubb), options={'ftol':1E-1})
##        myresults = optimize.basinhopping(fitfun_sumofsin_2angle2, p0, T=100, niter=200, niter_success=75, minimizer_kwargs=minimizer_kwargs, take_step=take_step)
##        print(myresults)
#        myresults = myresults.lowest_optimization_result
        
        print("Sum of Sines Order ",order, " --> Status: ", myresults.success," - ",myresults.status)
        print(myresults.message)
        print("Lowest cost:", myresults.cost)
        print("Parameters:",myresults.x)

       # Get the cost function
#        mycost[myrow] = myresults.cost
        mycost[myrow] = fitfun_sumofsin_2angle2(myresults.x, xxn[mysamp], xxb[mysamp], yy[mysamp])

        coeffs[order - 1, 0:myresults.x.size] = myresults.x
        
        xxn2 = np.linspace(np.min(xxn[mysamp]), np.max(xxn[mysamp]), 100)
        xxb2 = np.linspace(np.min(xxb[mysamp]), np.max(xxb[mysamp]), 100)
        mypred = fitfun_sumofsin_2angle(xxn2, xxb2, myresults.x)
        plt.plot(xxn2, mypred,label="order={0}".format(order))
    plt.legend(loc='upper center', bbox_to_anchor=(0.5,1.05), ncol=3, fancybox=True)

    fig = plt.figure(figsize=(7, 5), dpi=200)
    plt.plot(np.arange(1, myorder + 1), mycost)
    plt.xlabel('Sum of Sines Order')
    plt.ylabel('Cost function')
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight', dpi=200)

    fidx = mycost.argmin()
    print("Sum of Sines Order Selected: ", fidx + 1)

    return coeffs[fidx], fidx+1


def plot_bias(xx, dH, grp_xx, grp_dH, title, pp, pmod=None, smod=None, plotmin=None, txt=None):
    """
    data : original data as numpy array (:,2), x = 1col,y = 2col
    gdata : grouped data as numpy array (:,2)
    pmod,smod are two model options to plot, numpy array (:,1)
    """

    mykeep = np.isfinite(dH)
    xx = xx[mykeep]
    dH = dH[mykeep]
    if xx.size > 75000:
        mysamp = np.random.randint(0, xx.size, 75000)
    else:
        mysamp = np.arange(0, xx.size)
    # mysamp = mysamp.astype(np.int64) #huh?

    # title = 'Cross'
    fig = plt.figure(figsize=(7, 5), dpi=200)
    fig.suptitle(title + 'track bias', fontsize=14)
    if plotmin is None:
        plt.plot(xx[mysamp], dH[mysamp], '^', ms=0.2, color='0.5', rasterized=True, fillstyle='full')
        plt.plot(grp_xx, grp_dH, '-', ms=2, color='0.15')
    else:
        plt.plot(grp_xx, grp_dH, '^', ms=0.5, color='0.5', rasterized=True, fillstyle='full')

    if pmod is not None:
        plt.plot(grp_xx, pmod, 'r-', ms=2)
    if smod is not None:
        plt.plot(grp_xx, smod, 'm-', ms=2)

    plt.plot(grp_xx, np.zeros(grp_xx.size), 'k-', ms=1)

    plt.xlim(np.min(grp_xx), np.max(grp_xx))
    # plt.ylim(-200,200)
    ymin, ymax = plt.ylim((np.nanmean(dH[mysamp])) - 2 * np.nanstd(dH[mysamp]),
                          (np.nanmean(dH[mysamp])) + 2 * np.nanstd(dH[mysamp]))

    # plt.axis([0, 360, -200, 200])
    plt.xlabel(title + ' track distance [meters]')
    plt.ylabel('dH [meters]')
    plt.legend(('Raw [samples]', 'Grouped Median', 'Fit'), loc=1)

    if txt is not None:
        plt.text(0.05, 0.15, txt, fontsize=12, fontweight='bold', color='black',
                 family='monospace', transform=plt.gca().transAxes)

    # plt.show()
    pp.savefig(fig, bbox_inches='tight', dpi=200)


def correct_cross_track_bias(mst_dem, slv_dem, inang, pp, pts=False):
    # calculate along/across track coordinates
    myang = np.deg2rad(inang.img)
    xxr, yyr = get_xy_rot(slv_dem, myang)  # across,along track coordinates calculated from angle map

    # Calculate initial dH for comparison
    dH0 = calculate_dH(mst_dem, slv_dem, pts)

    # arrange the dependent (dH) and independent variables (ANGLE) into vectors
    # ALSO, filters the dH > threshold (40), and provides grouped statistics... 
    xx, dH, grp_xx, grp_dH = get_fit_variables(mst_dem, slv_dem, xxr, pts, pp, mytype='Cross')

    # #
    # Need conditional to check for large enough sample size... ?
    # #
    # POLYNOMIAL FITTING - here using my defined robust polynomial fitting
    pcoef, myorder = robust_polynomial_fit(grp_xx, grp_dH)
#    pcoef, myorder = robust_polynomial_fit(xx, dH)
    polymod = fitfun_polynomial(xx, pcoef)
    polymod_grp = fitfun_polynomial(grp_xx, pcoef)
    polyres = RMSE(dH - polymod)
    print("Cross track robust Polynomial RMSE (all data): ", polyres)

    #   # USING POLYFIT With GROUPED data
    #    pcoef, _ = polynomial_fit(grp_xx,grp_dH) #mean
#    pcoef2, _ = polynomial_fit(xx,dH) # USE ALL DATA
#    polymod=poly.polyval(xx,pcoef2)
#    polyres=RMSE(dH-polymod)
#    print("Cross track standard Polynomial RMSE (all data): ", polyres)

    # if pts:
    #    pcoef2, _ = polynomial_fit(xx,dH)
    #    polymod = poly.polyval(grp_xx,pcoef2)
    #    polyres = RMSE(grp_dH-polymod)
    #    print("Cross track Polynomial RMSE (raw data): ", polyres)
    mytext = "Polynomial order: " + np.str(myorder)
    plot_bias(xx, dH, grp_xx, grp_dH, 'Cross', pp, pmod=polymod_grp, txt=mytext)

    # Generate correction for DEM
#    out_corr = poly.polyval(xxr, pcoef)
    out_corr = fitfun_polynomial(xxr, pcoef)

    # Correct DEM
    zupdate = np.ma.array(slv_dem.img + out_corr, mask=slv_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)

    dH1 = calculate_dH(mst_dem, slv_dem, pts)

    if not pts:
        final_histogram(dH0.img, dH1.img, pp)
    elif pts:
        final_histogram(dH0, dH1, pp)

    return slv_dem, out_corr, pcoef


def correct_along_track_bias(mst_dem, slv_dem, inang, pp, pts):
    # calculate along/across track coordinates
    xxr, yyr = get_xy_rot(slv_dem, np.deg2rad(inang.img))  # across,along track coordinates calculated from angle map

    # Calculate initial dH for comparison
    dH0 = calculate_dH(mst_dem, slv_dem, pts)

    # arrange the dependent (dH) and independent variables (ANGLE) into vectors
    # ALSO, filters the dH > threshold (40), and provides grouped statistics... 
    xx, dH, grp_xx, grp_dH = get_fit_variables(mst_dem, slv_dem, yyr, pts, pp, mytype='Along')

    #    # POLYNOMIAL FITTING - here using my defined robust polynomial fitting
    #    pcoef, _ = robust_polynomial_fit(xx,dH)
    #    polymod=fitfun_polynomial(xx,pcoef)
    #    polymod_grp=fitfun_polynomial(grp_xx,pcoef)
    #    polyres=RMSE(dH-polymod)
    #    print("Along track Polynomial RMSE (all data): ", polyres)

    #   # USING POLYFIT With GROUPED data
    #    pcoef, _ = polynomial_fit(grp_xx,grp_dH) #mean
    #    #pcoef2, _ = polynomial_fit(xx,dH) # USE ALL DATA
    #    polymod=poly.polyval(grp_xx,pcoef)
    #    polyres=RMSE(grp_dH-polymod)
    #    print("Along track Polynomial RMSE (grouped data): ", polyres)

    # SUM OF SINES
    # First define the bounds of the three sine wave coefficients to solve
#    lb = np.asarray([2, np.divide(2 * np.pi, 300000), 0])
#    ub = np.asarray([15, np.divide(2 * np.pi, 20000), 2*np.pi])
    lb = np.asarray([1, 15, 0])
    ub = np.asarray([15, 100, 2*np.pi])
    scoef, myorder = function_sum_of_sin(grp_xx, grp_dH, lb, ub, pp)
#    scoef, myorder = function_sum_of_sin(xx,dH, lb, ub, pp)
    scoef = np.trim_zeros(scoef,'b')
    sinmod = fitfun_sumofsin(xx, scoef)
    sinmod_grp = fitfun_sumofsin(grp_xx, scoef)
    
    sinres = RMSE(dH - sinmod)
    sinres_grp = RMSE(grp_dH - sinmod_grp)
    print("Along track Sum_of_Sin RMSE (grouped): ", sinres_grp)
    print("Along track Sum_of_Sin RMSE (all data): ", sinres)

    mytext = "Sum of " + np.str(myorder) + " sines"
    plot_bias(xx, dH, grp_xx, grp_dH, 'Along', pp, smod=sinmod_grp, txt=mytext)

    # ADD CONDITIONAL FOR CHOOSING WHICH FIT
    #    out_corr = fitfun_polynomial(xx,pcoef)
    out_corr2 = fitfun_sumofsin(xx, scoef)

    if not pts:
        # res1 = RMSE(dH-out_corr)
        res2 = RMSE(dH - out_corr2)
        # print("ALL Pixels/Points Polynomial RMSE:", res1)
        print("ALL Pixels/Points Sum_of_Sin RMSE:", res2)

    #    if sinres<=polyres:
    mycorr = fitfun_sumofsin(yyr, scoef)
    #    elif polyres<sinres:
    #        mycorr = poly.polyval(yyr,pcoef)
    #
    #    NEVER GOT TH EMASK THING WORKING...
    zupdate = np.ma.array(slv_dem.img + mycorr, mask=slv_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)

    dH1 = calculate_dH(mst_dem, slv_dem, pts)

    if not pts:
        final_histogram(dH0.img, dH1.img, pp)
    elif pts:
        final_histogram(dH0, dH1, pp)

    return slv_dem, mycorr, scoef


def correct_along_track_jitter(mst_dem, slv_dem, inangN, inangB, pp, pts):
    
    # calculate along/across track coordinates
    # myang = np.deg2rad(np.multiply(inang,np.multiply(dH,0)+1))# generate synthetic angle image for testing
    myangN = np.deg2rad(inangN.img)
    myangB = np.deg2rad(inangB.img)
    xxn_mat, xxb_mat = get_atrack_coord(slv_dem, myangN, myangB)  # across,along track coordinates calculated from angle map

    # Calculate initial dH for comparison
    dH0 = calculate_dH(mst_dem, slv_dem, pts)

    # arrange the dependent (dH) and independent variables (ANGLE) into vectors
    # ALSO, filters the dH > threshold (40), and provides grouped statistics... 
    if not pts:
        xxn, xxb, dH, grp_xx, grp_dH = get_fit_variables(mst_dem, slv_dem, xxn_mat, pts, pp, xxb=xxb_mat,
                                                         mytype='Jitter')
    elif pts:
        xxn, xxb, dH, grp_xx, grp_dH = get_fit_variables(mst_dem, slv_dem, xxn_mat, pts, pp, xxb=xxb_mat,
                                                         mytype='Jitter')
    # #
    # Need conditional to check for enough sample size... 
    # #

    # SUM OF SINES
    # First define the bounds of the three sine wave coefficients to solve
    lb = np.asarray([0.3, 3, 0, 0.3, 3, 0])
    ub = [10, 100, 2*np.pi, 10, 100, 2*np.pi]

    #    xxn_vec = np.reshape(xxn,(xxn.size,1))
    #    xxb_vec = np.reshape(xxb,(xxb.size,1))
    #    dH_vec = np.reshape(dH.img,(dH.img.size,1))

    scoef, myorder = function_sum_of_sin_2angle(xxn, xxb, dH, lb, ub, pp, ylim=300)
    scoef = np.trim_zeros(scoef,'b')
    print(scoef)
    sinmod = fitfun_sumofsin_2angle(xxn, xxb, scoef)
    sinmod2 = fitfun_sumofsin_2angle(np.linspace(np.nanmin(xxn), np.nanmax(xxn), grp_xx.size),
                                     np.linspace(np.nanmin(xxn), np.nanmax(xxn), grp_xx.size), scoef)
    # embed()
    sinres = RMSE(dH - sinmod)
    print("Along track Sum_of_Sin RMSE: ", sinres)

    # plot_bias(orig_data,grp_data,mytype,pp)
    mytext = "Sum of " + np.str(myorder) + "(x2) sines"
    plot_bias(xxn, dH, grp_xx, grp_dH, 'Jitter', pp, smod=sinmod2, txt=mytext)

    if not pts:
        out_corr = np.reshape(sinmod, slv_dem.img.shape)
    elif pts:
        sinmod_mat = fitfun_sumofsin_2angle(xxn_mat, xxb_mat, scoef)
        out_corr = np.reshape(sinmod_mat, slv_dem.img.shape)

    fig = plt.figure(figsize=(7, 5), dpi=200)
    ax = plt.gca()
    # fig.suptitle(title, fontsize = 14)
    plt1 = plt.imshow(out_corr)
    plt1.set_clim(np.nanmin(out_corr), np.nanmax(out_corr))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(plt1, cax=cax)
    plt.tight_layout()
    pp.savefig(fig, bbox_inches='tight', dpi=200)
    zupdate = np.ma.array(slv_dem.img + out_corr, mask=slv_dem.img.mask)  # shift in z
    slv_dem = slv_dem.copy(new_raster=zupdate)

    dH1 = calculate_dH(mst_dem, slv_dem, pts)
    if not pts:
        false_hillshade(dH1, 'Post-Jitter Removal', pp)
        final_histogram(dH0.img, dH1.img, pp)
    elif pts:
        final_histogram(dH0, dH1, pp)

    return slv_dem, out_corr, scoef

################################################################################################
# the big kahuna
def mmaster_bias_removal(mst_dem, slv_dem, glacmask=None, landmask=None, pts=False, out_dir='.'):
    """
    Removes cross track and along track biases from MMASTER DEMs. 

    Parameters
    ----------
    mst_DEM : string or GeoImg
        Path to filename or GeoImg dataset representing "master" DEM or ICESat.
    slv_DEM : string or GeoImg
        Path to filename or GeoImg dataset representing "slave" DEM (developed for ASTER).
    glacmask : string, optional
        Path to shapefile representing points to exclude from co-registration
        consideration (i.e., glaciers).
    landmask : string, optional
        Path to shapefile representing points to include in co-registration
        consideration (i.e., stable ground/land).
    pts : bool, optional
        If True, program assumes that masterDEM represents point data (i.e., ICESat),
        as opposed to raster data. Slope/aspect are then calculated from slaveDEM.
        masterDEM should be a string representing an HDF5 file continaing ICESat data.
    out_dir : string, optional
        Location to save bias removal outputs. [default to the current directory]
    """

    # if the output directory does not exist, create it.
    out_dir = os.path.abspath(out_dir)
    try:
        os.makedirs(out_dir)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(out_dir):
            pass
        else:
            raise
    print(out_dir)

    # import angle data
    ang_mapN = GeoImg('TrackAngleMap_3N.tif')
    ang_mapB = GeoImg('TrackAngleMap_3B.tif')
    ang_NB = np.array(np.divide(ang_mapN.img + ang_mapB.img, 2))  # shift in z
    ang_mapNB = ang_mapN.copy(new_raster=ang_NB)

    # pre-processing steps (co-registration,Correlation_masking)
    mst_coreg, slv_coreg, shift_params = preprocess(mst_dem, slv_dem, glacmask=glacmask, landmask=landmask,
                                                    cwd='.', out_dir=out_dir, pts=pts)

    # OPEN and start the Results.pdf
    pp = PdfPages(os.path.sep.join([out_dir, 'BiasCorrections_Results.pdf']))

    ### Create the stable land masks
    stable_mask = create_stable_mask(slv_coreg, glacmask, landmask)
    fmaskpoly = get_aster_footprint(slv_dem.rsplit('_Z.tif')[0],epsg=slv_coreg.epsg)
    fmask = make_mask(fmaskpoly,slv_coreg, raster_out=True)
    
    ### PREPARE numpy masked arrays for .img data
    slv_coreg.unmask()
#    slv_coreg.mask(np.logical_or(np.invert(fmask),stable_mask))
    slv_coreg.mask(np.logical_or.reduce((np.invert(fmask),stable_mask,np.isnan(slv_coreg.img))))

    # cross-track bias removal 
    # - assumes both dems include only stable terrain.
    # - Errors permitted as we will filter along the way
    slv_coreg_xcorr, xcorr, pcoef = correct_cross_track_bias(mst_coreg, slv_coreg, ang_mapN, pp, pts=pts)
    outname = os.path.splitext(slv_dem)[0] + "_adj_X.tif"
    slv_coreg_xcorr.write(outname, out_folder=out_dir)
    np.savetxt(os.path.sep.join([out_dir, 'params_CrossTrack_Polynomial.txt']), pcoef)
    plt.close("all")

    ### PREPARE numpy masked arrays for .img data
    slv_coreg_xcorr.unmask()
#    slv_coreg_xcorr.mask(np.logical_or(np.invert(fmask),stable_mask))
    slv_coreg_xcorr.mask(np.logical_or.reduce((np.invert(fmask),stable_mask,np.isnan(slv_coreg.img))))
    
    # along-track bias removal
    slv_coreg_xcorr_acorr, acorr, scoef = correct_along_track_bias(mst_coreg, slv_coreg_xcorr, ang_mapNB, pp, pts=pts)
    outname = os.path.splitext(slv_dem)[0] + "_adj_XA.tif"
    slv_coreg_xcorr_acorr.write(outname, out_folder=out_dir)
    np.savetxt(os.path.sep.join([out_dir, 'params_AlongTrack_SumofSines.txt']), scoef)
    plt.close("all")

    ### PREPARE numpy masked arrays for .img data
    slv_coreg_xcorr_acorr.unmask()
#    slv_coreg_xcorr_acorr.mask(np.logical_or(np.invert(fmask),stable_mask))
    slv_coreg_xcorr_acorr.mask(np.logical_or.reduce((np.invert(fmask),stable_mask,np.isnan(slv_coreg.img))))
    
    # along-track jitter removal
    # slv_coreg_xcorr_acorr_jcorr, jcorr = correct_along_track_jitter(mst_coreg,slv_coreg_xcorr_acorr,ang_mapNB,pp)
    slv_coreg_xcorr_acorr_jcorr, jcorr, scoef2 = correct_along_track_jitter(mst_coreg, slv_coreg_xcorr_acorr, ang_mapN,
                                                                             ang_mapB, pp, pts=pts)
    outname = os.path.splitext(slv_dem)[0] + "_adj_XAJ.tif"
    slv_coreg_xcorr_acorr_jcorr.write(outname, out_folder=out_dir)
    np.savetxt(os.path.sep.join([out_dir, 'params_AlongTrackJitter_SumofSines.txt']), scoef2)
    plt.close("all")

    ### PREPARE numpy masked arrays for .img data
    slv_coreg_xcorr_acorr_jcorr.unmask()
#    slv_coreg_xcorr_acorr_jcorr.mask(np.logical_or(np.invert(fmask),stable_mask))
    slv_coreg_xcorr_acorr_jcorr.mask(np.logical_or.reduce((np.invert(fmask),stable_mask,np.isnan(slv_coreg.img))))
    
    # EXPERIMENTAL along-track jitter removal - 2nd iteration
    # slv_coreg_xcorr_acorr_jcorr, jcorr = correct_along_track_jitter(mst_coreg,slv_coreg_xcorr_acorr,ang_mapNB,pp)
#    slv_coreg_xcorr_acorr_jcorr2, jcorr2, scoef3 = correct_along_track_jitter(mst_coreg, slv_coreg_xcorr_acorr_jcorr,
#                                                                               ang_mapN, ang_mapB, pp, pts=pts)
#    outname = os.path.splitext(slv_dem)[0] + "_adj_XAJJ.tif"
#    slv_coreg_xcorr_acorr_jcorr2.write(outname, out_folder=out_dir)
#    np.savetxt(os.path.sep.join([out_dir, 'params_AlongTrackJitter_SumofSines2.txt']), scoef3)
#    plt.close("all")
#
    dH0 = calculate_dH(mst_coreg, slv_coreg, pts)
    dH_final = calculate_dH(mst_coreg, slv_coreg_xcorr_acorr_jcorr, pts)

    if not pts:
        # Calculate initial differences
        mytitle = 'dH Initial'
        false_hillshade(dH0, mytitle, pp)

        # Calculate post correction differences
        mytitle = 'dH Post corrections'
        false_hillshade(dH_final, mytitle, pp)

        final_histogram(dH0.img, dH_final.img, pp)
    elif pts:
        # Calculate initial differences
        final_histogram(dH0, dH_final, pp)

    #
    # re-coregister
    print('Re-co-registering DEMs.')
    recoreg_outdir = os.path.sep.join([out_dir, 're-coreg'])
    if not pts:
        mst_coreg, slv_adj_coreg, shift_params2 = dem_coregistration(mst_coreg, slv_coreg_xcorr_acorr_jcorr,
                                                                     glaciermask=glacmask, landmask=landmask,
                                                                     outdir=recoreg_outdir, pts=pts)
    elif pts:
        mst_coreg, slv_adj_coreg, shift_params2 = dem_coregistration(mst_dem, slv_coreg_xcorr_acorr_jcorr,
                                                                     glaciermask=glacmask, landmask=landmask,
                                                                     outdir=recoreg_outdir, pts=pts)

    plt.close("all")
    # clean-up 
    pp.close()
    print("Fin.")

    return mst_coreg, slv_coreg_xcorr_acorr_jcorr
