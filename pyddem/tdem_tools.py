"""
pymmaster.tdem_tools provides tools to post-process DEM stacks: volume integration over specific outlines, comparison to point data, spatial aggregation...
"""
import xarray as xr
import os
import sys
import shutil
import numpy as np
from itertools import groupby
from operator import itemgetter
import multiprocessing as mp
from scipy.interpolate import interp1d
import gdal
import osr
import ogr
import pyproj
import time
from datetime import datetime as dt
import pandas as pd
import pyddem.stack_tools as st
import pyddem.vector_tools as vt
import pyddem.fit_tools as ft
import pyddem.spstats_tools as spt
import pyddem.volint_tools as volt
from pybob.coreg_tools import get_slope, create_stable_mask, dem_coregistration
from pybob.GeoImg import GeoImg
from pybob.ICESat import ICESat
from glob import glob
import itertools

def inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_field_name):
    # get intersecting rgiid for each stack extent
    list_list_rgiid = []
    for fn_stack in list_fn_stack:
        #TODO: temporary test
        # ds = xr.open_dataset(fn_stack)
        # extent, proj = st.extent_stack(ds)
        ds = xr.open_dataset(fn_stack)
        tile_name = st.tilename_stack(ds)
        lat, lon = vt.SRTMGL1_naming_to_latlon(tile_name)
        extent = [lon, lat, lon+1, lat+1]
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)

        list_rgiid = vt.list_shp_field_inters_extent(fn_shp, feat_field_name, extent, proj.ExportToWkt())
        list_list_rgiid.append(list_rgiid)

    # get all rgiids intersecting all stacks without duplicates
    all_rgiid = []
    for list_rgiid in list_list_rgiid:
        all_rgiid = all_rgiid + list_rgiid
    all_rgiid = list(set(all_rgiid))

    # inverting to have intersecting stacks by rgiid: more practical to compute that way
    list_list_stack_by_rgiid = []
    for rgiid in all_rgiid:
        list_stacks = []
        for fn_stack in list_fn_stack:
            if rgiid in list_list_rgiid[list_fn_stack.index(fn_stack)]:
                list_stacks.append(fn_stack)
        list_list_stack_by_rgiid.append(sorted(list_stacks))

    return all_rgiid, list_list_stack_by_rgiid


def sel_dc(ds, tlim, mask):
    # select data cube on temporal and spatial mask
    if tlim is None:
        time1 = time2 = None
    else:
        time1, time2 = tlim

    index = np.where(mask)
    minx = ds.x[np.min(index[1])].values
    maxx = ds.x[np.max(index[1])].values
    miny = ds.y[np.min(index[0])].values
    maxy = ds.y[np.max(index[0])].values

    dc = ds.sel(dict(time=slice(time1, time2), x=slice(minx, maxx), y=slice(miny, maxy)))
    submask = mask[np.min(index[0]):np.max(index[0] + 1), np.min(index[1]):np.max(index[1] + 1)]

    return dc, submask, (slice(minx,maxx),slice(miny,maxy))

def int_dc(dc, mask, **kwargs):
    # integrate data cube over masked area
    dh = dc.variables['z'].values - dc.variables['z'].values[0]
    err = np.sqrt(dc.variables['z_ci'].values ** 2 + dc.variables['z_ci'].values[0] ** 2)
    ref_elev = dc.variables['z'].values[0]
    slope = get_slope(st.make_geoimg(dc, 0)).img
    slope[np.logical_or(~np.isfinite(slope), slope > 70)] = np.nan

    t, y, x = np.shape(dh)
    dx = np.round((dc.x.max().values - dc.x.min().values) / float(x))

    df = volt.hypso_dc(dh,err,ref_elev,mask,dx,**kwargs)

    return df

def get_inters_stack_dem(list_fn_stack,ext,proj):

    # get stacks that intersect the extent
    poly = vt.poly_from_extent(ext)

    list_inters_stack = []
    for fn_stack in list_fn_stack:
        ds = xr.open_dataset(fn_stack)
        ext_st, proj_st = st.extent_stack(ds)
        poly_st = vt.poly_from_extent(ext_st)
        trans = vt.coord_trans(True, proj_st, True, proj)
        poly_st.Transform(trans)

        if poly_st.Intersect(poly):
            list_inters_stack.append(fn_stack)

    return list_inters_stack

def comp_stacks_dem(list_fn_stack,fn_dem,inc_mask=None, exc_mask=None, get_timelapse_filtstack=False, outfile=None):

    #get list of intersecting stacks
    ext, proj = vt.extent_rast(fn_dem)
    list_inters_stack = get_inters_stack_dem(list_fn_stack,ext,proj)
    dem_full = GeoImg(fn_dem)

    list_dh = list_z_score = list_dt = []
    for fn_stack in list_inters_stack:

        ds = xr.open_dataset(fn_stack)
        ref_img = st.make_geoimg(ds)
        mask = create_stable_mask(ref_img,exc_mask,inc_mask)

        dem = dem_full.reproject(ref_img)

        x,y = np.where(mask)
        t = np.array([dem.datetime]*len(x))

        dem_sub = dem[x,y]
        comp = ds.interp(time=t,x=x,y=y)
        comp_h = comp.variables['z'].values
        comp_ci = comp.variables['z_ci'].values

        dh = dem_sub - comp_h
        z_score = dh / comp_ci

        list_dh.append(dh)
        list_z_score.append(z_score)

        if get_timelapse_filtstack:
            #here we assume filtered stack is stored at the same location
            fn_filt = os.path.join(os.path.dirname(fn_stack),os.path.basename(fn_stack).split('_')[0]+'_filtered.nc')
            ds_filt = xr.open_dataset(fn_filt)

            ds_near = ds_filt.isel(time=t,x=x,y=y,method='nearest')
            near_times = ds_near.time.values
            delta_t = near_times - t

            list_dt.append(delta_t)

    full_dh = np.concatenate(list_dh)
    full_z_score = np.concatenate(list_z_score)
    full_dt = np.concatenate(list_dt)

    df = pd.DataFrame()
    df = df.assign(dh=full_dh,z_score=full_z_score,dt=full_dt)

    if outfile is None:
        return df
    else:
        df.to_csv(outfile)

def datetime_to_yearfraction(date):

    #ref:https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years
    def sinceEpoch(date): #returns seconds since epoch

        return time.mktime(date.timetuple())

    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction


def create_tmp_dir_for_outfile(file_out):

    tmp_dir = os.path.join(os.path.dirname(file_out), 'tmp_'+os.path.splitext(os.path.basename(file_out))[0]) + os.path.sep
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    return tmp_dir

def remove_tmp_dir_for_outfile(file_out):

    tmp_dir = os.path.join(os.path.dirname(file_out), 'tmp_'+os.path.splitext(os.path.basename(file_out))[0]) + os.path.sep
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir,ignore_errors=True)

def saga_aspect_slope_curvature(dem_in, topo_out, method_nb=5):
    """
    :param dem_in: dem input
    :param topo_out: raster out (3 bands: aspect, slope, curvature)
    :param method_nb: algorithm used, see function description
    :return:

    requirement: SAGA 2.X with X>3

    #ref:http://www.saga-gis.org/saga_tool_doc/2.2.3/ta_morphometry_0.html

    aspect, slope, curvature methods
    methods number
    [0] maximum slope (Travis et al. 1975)
    [1] maximum triangle slope (Tarboton 1997)
    [2] least squares fitted plane (Horn 1981, Costa-Cabral & Burgess 1996)
    [3] 6 parameter 2nd order polynom (Evans 1979)
    [4] 6 parameter 2nd order polynom (Heerdegen & Beran 1982)
    [5] 6 parameter 2nd order polynom (Bauer, Rohdenburg, Bork 1985)
    [6] 9 parameter 2nd order polynom (Zevenbergen & Thorne 1987)
    [7] 10 parameter 3rd order polynom (Haralick 1983)

    unit slope 0=rad, 1=deg, 2=percent
    unit aspect 0=rad, 1=deg
    """

    tmp_dir = create_tmp_dir_for_outfile(topo_out)

    saga_elev = tmp_dir + 'elev_temp.sgrd'
    saga_slope = tmp_dir + 'slope_temp.sgrd'
    saga_aspect = tmp_dir + 'aspect_temp.sgrd'
    saga_max_curv = tmp_dir + 'max_curv_temp.sgrd'
    tif_slope = tmp_dir + 'slope_temp.tif'
    tif_aspect = tmp_dir + 'aspect_temp.tif'
    tif_max_curv = tmp_dir + 'max_curv_temp.tif'
    output_vrt = tmp_dir + 'stack.vrt'

    os.system('saga_cmd io_gdal 0 -GRIDS ' + saga_elev + ' -FILES ' + dem_in)
    os.system(
        'saga_cmd ta_morphometry 0 -ELEVATION ' + saga_elev + ' -SLOPE ' + saga_slope + ' -ASPECT ' + saga_aspect + ' -C_MAXI ' + saga_max_curv + ' -METHOD ' + str(
            method_nb) + ' -UNIT_SLOPE 1 -UNIT_ASPECT 1')
    os.system('saga_cmd io_gdal 2 -GRIDS ' + saga_slope + ' -FILE ' + tif_slope)
    os.system('saga_cmd io_gdal 2 -GRIDS ' + saga_aspect + ' -FILE ' + tif_aspect)
    os.system('saga_cmd io_gdal 2 -GRIDS ' + saga_max_curv + ' -FILE ' + tif_max_curv)

    os.system(
        'gdalbuildvrt -separate -overwrite ' + output_vrt + ' ' + tif_slope + ' ' + tif_aspect + ' ' + tif_max_curv)
    os.system('gdal_translate ' + output_vrt + ' ' + topo_out)

    remove_tmp_dir_for_outfile(topo_out)


def icesat_comp_wrapper(argsin):

    fn_stack,ice_coords,ice_latlon,ice_elev,ice_date,groups,dates,read_filt,gla_mask,inc_mask,exc_mask = argsin

    full_h, full_dh, full_z_score, full_dt, full_pos, full_slp, full_lat, full_lon, full_dh_ref, full_curv, full_dh_tot, full_time = ([] for i in range(12))
    # full_time = np.array([],dtype='datetime64[D]')
    ds = xr.open_dataset(fn_stack)
    ds.load()
    tile_name = st.tilename_stack(ds)

    tmp_h = st.make_geoimg(ds)
    tmp_ci = st.make_geoimg(ds)
    tmp_dhtot = st.make_geoimg(ds)
    tmp_slope = st.make_geoimg(ds)
    tmp_slope.img = ds.slope.values
    tmp_dhtot.img = ds.z[-1, :].values - ds.z[0, :].values

    # tmp_slope.img = np.ones(np.shape(tmp_slope.img))*20

    ds_sub = ds.interp(time=dates)

    terrain_mask = None
    get_curv = True

    #glacier mask has priority and won't be changed, gla = 2
    if gla_mask is not None:

        print('Tile '+tile_name+': deriving terrain glacier mask...')
        #rasterizing mask and changing to float to use geoimg.raster_points2
        mask = ft.get_stack_mask(gla_mask,ds)
        tmp_exc_mask = st.make_geoimg(ds)
        tmp_exc_mask.img = np.zeros(np.shape(mask),dtype='float32')
        tmp_exc_mask.img[mask] = 2

        terrain_mask = tmp_exc_mask

    #here we take the difference of inclusion and exclusion mask for the rest of terrain, inc = 1, exc = 0
    if inc_mask is not None:
        print('Tile ' + tile_name + ': deriving terrain inclusion mask...')
        # rasterizing mask and changing to float to use geoimg.raster_points2
        mask = ft.get_stack_mask(inc_mask, ds)

        if terrain_mask is not None:
            land_mask = np.logical_and(terrain_mask.img == 0, mask)
            terrain_mask.img[land_mask] = 1
        else:
            tmp_inc_mask = st.make_geoimg(ds)
            tmp_inc_mask.img = np.zeros(np.shape(mask),dtype='float32')
            tmp_inc_mask[mask] = 1

            terrain_mask = tmp_inc_mask

    if exc_mask is not None:
        print('Tile ' + tile_name + ': deriving terrain exclusion mask...')
        # rasterizing mask and changing to float to use geoimg.raster_points2
        mask = ft.get_stack_mask(exc_mask, ds)

        if terrain_mask is not None:
            noland_mask = np.logical_and(terrain_mask.img <= 1, mask)
            terrain_mask.img[noland_mask] = 0
        else:
            tmp_exc_mask = st.make_geoimg(ds)
            tmp_exc_mask.img = np.ones(np.shape(mask), dtype='float32')
            tmp_exc_mask[mask] = 0

            terrain_mask = tmp_exc_mask

    fn_raw = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '.nc')
    ds_raw = xr.open_dataset(fn_raw)
    tmp_ref_h = st.make_geoimg(ds_raw,band=(slice(0,None),slice(0,None)),var='ref_z')
    tmp_ref_h = tmp_ref_h.reproject(tmp_h)

    #get curvature
    if get_curv:
        fn_tmp_ref = os.path.join(os.path.dirname(fn_stack),'tmp_ref_'+os.path.basename(fn_stack).split('_')[0]+'.tif')
        tmp_ref_h.write(fn_tmp_ref)
        fn_tmp_curv = os.path.join(os.path.dirname(fn_stack),'tmp_curv_'+os.path.basename(fn_stack).split('_')[0]+'.tif')
        saga_aspect_slope_curvature(fn_tmp_ref,fn_tmp_curv)
        ds_curv = gdal.Open(fn_tmp_curv,gdal.GA_ReadOnly)
        curv_arr = ds_curv.GetRasterBand(3).ReadAsArray()
        tmp_curv = GeoImg(fn_tmp_curv)
        tmp_curv.img = curv_arr
        ds_curv = None
        os.remove(fn_tmp_curv)
        os.remove(fn_tmp_ref)
    else:
        tmp_curv = st.make_geoimg(ds)
        tmp_curv.img = np.zeros(np.shape(tmp_curv.img),dtype='float32')

    if read_filt:
        print('Tile '+tile_name+': getting original data dates from filtered array...')

        #we read the boolean data cube indicating positions where original data was used
        tmp_dt = st.make_geoimg(ds)
        fn_filt = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '_filtered.nc')
        ds_filt = xr.open_dataset(fn_filt)
        ds_filt.load()

        #first, remove duplicate dates by merging boolean arrays for same dates
        t_vals = list(ds_filt.time.values)
        dates_rm_dupli = sorted(list(set(t_vals)))
        ind_firstdate = []
        for i, date in enumerate(dates_rm_dupli):
            ind_firstdate.append(t_vals.index(date))
        ds_filt2 = ds_filt.isel(time=np.array(ind_firstdate))
        for i in range(len(dates_rm_dupli)):
            t_ind = (t_vals == dates_rm_dupli[i])
            if len(t_ind) > 1:
                ds_filt2.z.values[i, :] = np.any(ds_filt.z[t_vals == dates_rm_dupli[i], :].values.astype(bool), axis=0)

            # getting time data as days since 2000
        y0 = np.datetime64('2000-01-01')
        ftime = ds_filt2.time.values
        ftime_delta = np.array([t - y0 for t in ftime])
        days = [td.astype('timedelta64[D]').astype(int) for td in ftime_delta]

        # reindex to get closest not-NaN time value of the date vector
        filt_arr = np.array(ds_filt2.z.values, dtype='float32')
        filt_arr[filt_arr == 0] = np.nan
        days = np.array(days)
        filt_arr = filt_arr * days[:, None, None]
        at_least_2 = np.count_nonzero(~np.isnan(filt_arr), axis=0) >= 2
        filt_tmp = filt_arr[:, at_least_2]
        out_arr = np.copy(filt_tmp)
        for i in range(np.shape(filt_tmp)[1]):
            ind = ~np.isnan(filt_tmp[:, i])
            fn = interp1d(days[ind], filt_tmp[:, i][ind], kind='nearest', fill_value='extrapolate', assume_sorted=True)
            out_arr[:, i] = fn(days)
        filt_arr[:, at_least_2] = out_arr
        ds_filt2.z.values = filt_arr
        ds_filt_sub = ds_filt2.reindex(time=dates, method='nearest')


    for i, group in enumerate(groups):
        #keep only a campaign group
        pts_idx_dt = np.array(ice_date == group)
        date = dates[i]

        print('Tile '+tile_name+': calculating differences for the ' + str(
            np.count_nonzero(pts_idx_dt)) + ' points of campaign:' + str(date))

        subsamp_ice = [tup for i, tup in enumerate(ice_coords) if pts_idx_dt[i]]
        subsamp_latlon = [tup for i, tup in enumerate(ice_latlon) if pts_idx_dt[i]]

        tmp_h.img = ds_sub.z[i, :].values
        comp_pts_h = tmp_h.raster_points(subsamp_ice, nsize=3, mode='linear')

        tmp_ci.img = ds_sub.z_ci[i, :].values
        comp_pts_ci = tmp_ci.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_dhtot = tmp_dhtot.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_slope = tmp_slope.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_ref_h = tmp_ref_h.raster_points(subsamp_ice, nsize=3, mode='linear')

        comp_pts_curv = tmp_curv.raster_points(subsamp_ice,nsize=3,mode='linear')

        dh = ice_elev[pts_idx_dt] - comp_pts_h
        good_vals = np.isfinite(dh)
        dh = dh[good_vals]
        h = ice_elev[pts_idx_dt][good_vals]
        z_score = dh / comp_pts_ci[good_vals]
        slp = comp_pts_slope[good_vals]
        dh_ref = ice_elev[pts_idx_dt] - comp_pts_ref_h
        dh_ref = dh_ref[good_vals]
        curv = comp_pts_curv[good_vals]
        dh_tot = comp_pts_dhtot[good_vals]

        if read_filt:
            day_diff = (date - y0).astype('timedelta64[D]').astype(int)
            tmp_dt.img = np.abs(ds_filt_sub.z[i, :].values - np.ones(np.shape(tmp_dt.img)) * day_diff)
            comp_pts_dt = tmp_dt.raster_points(subsamp_ice, nsize=3, mode='mean')
            dt_out = comp_pts_dt[good_vals]
        else:
            dt_out = np.zeros(len(dh)) * np.nan

        if terrain_mask is not None:
            comp_pts_mask = terrain_mask.raster_points(subsamp_ice, nsize=5, mode='nearest')
            pos = comp_pts_mask[good_vals]
        else:
            pos = np.zeros(len(dh))


        good_subsamp_latlon = [tup for i, tup in enumerate(subsamp_latlon) if good_vals[i]]
        if len(good_subsamp_latlon) > 0:
            lon = np.array([tup[1] for i, tup in enumerate(good_subsamp_latlon)])
            lat = np.array([tup[0] for i, tup in enumerate(good_subsamp_latlon)])
        else:
            lat=np.array([])
            lon=np.array([])

        full_h.append(h)
        full_dh.append(dh)
        full_z_score.append(z_score)
        full_dt.append(dt_out)
        full_pos.append(pos)
        full_slp.append(slp)
        full_time.append(np.array([date]*len(dh),dtype='datetime64[D]'))
        full_curv.append(curv)
        full_lat.append(lat)
        full_lon.append(lon)
        full_dh_ref.append(dh_ref)
        full_dh_tot.append(dh_tot)

    full_h = np.concatenate(full_h)
    full_dh = np.concatenate(full_dh)
    full_z_score = np.concatenate(full_z_score)
    full_dt = np.concatenate(full_dt)
    full_pos = np.concatenate(full_pos)
    full_slp = np.concatenate(full_slp)
    full_time = np.concatenate(full_time)
    full_lat = np.concatenate(full_lat)
    full_lon = np.concatenate(full_lon)
    full_dh_ref = np.concatenate(full_dh_ref)
    full_curv = np.concatenate(full_curv)
    full_dh_tot = np.concatenate(full_dh_tot)


    return full_h, full_dh, full_z_score, full_dt, full_pos, full_slp, full_time, full_lon, full_lat, full_dh_ref, full_curv, full_dh_tot


def comp_stacks_icesat(list_fn_stack,fn_icesat,gla_mask=None,inc_mask=None,exc_mask=None,nproc=1,read_filt=False,shift=None):

    ice = ICESat(fn_icesat)
    el_limit = -200
    mykeep = ice.elev > el_limit

    #exception for North Asia subreg 3 ICESat file
    if os.path.basename(fn_icesat) == 'ICESat_10_3_rgi50_NorthAsia.h5':
        keep_west = np.logical_and(ice.lon<-172,ice.lon>-179.999)
        keep_east_1 = np.logical_and(ice.lon<96,ice.lon>88)
        keep_east_2 = np.logical_and(ice.lon<146.999,ice.lon>138)
        keep_northasia_3 = np.logical_or.reduce((keep_west,keep_east_1,keep_east_2))
        mykeep = np.logical_and(mykeep,keep_northasia_3)

    ice.x = ice.x[mykeep]
    ice.y = ice.y[mykeep]
    ice.lat = ice.lat[mykeep]
    ice.lon = ice.lon[mykeep]
    ice.elev = ice.elev[mykeep]
    ice.UTCTime = ice.UTCTime[mykeep]
    ice.xy = list(zip(ice.x, ice.y))

    #putting code above in ICESat.clean() breaks co-registration...
    # ice.clean(el_limit=-200)

    #get intersecting tiles
    bounds = ice.get_bounds()
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    proj_wkt = proj.ExportToWkt()
    list_inters = get_inters_stack_dem(list_fn_stack,[bounds[0],bounds[2],bounds[1],bounds[3]],proj_wkt)

    if len(list_inters) == 0:
        print('Found no intersection to ICESat file: '+fn_icesat)
        raise ValueError('No intersecting file!')

    #laser campaigns of ICESat
    laser_op_icesat = [(dt(2003, 2, 20), dt(2003, 3, 29)), (dt(2003, 9, 25), dt(2003, 11, 19)),
                       (dt(2004, 2, 17), dt(2004, 3, 21)),
                       (dt(2004, 5, 18), dt(2004, 6, 21)), (dt(2004, 10, 3), dt(2004, 11, 8)),
                       (dt(2005, 2, 17), dt(2005, 3, 24)),
                       (dt(2005, 5, 20), dt(2005, 6, 23)), (dt(2005, 10, 21), dt(2005, 11, 24)),
                       (dt(2006, 2, 22), dt(2006, 3, 28)),
                       (dt(2006, 5, 24), dt(2006, 6, 26)), (dt(2006, 10, 25), dt(2006, 11, 27)),
                       (dt(2007, 3, 12), dt(2007, 4, 14)),
                       (dt(2007, 10, 2), dt(2007, 11, 5)), (dt(2008, 2, 17), dt(2008, 3, 21)),
                       (dt(2008, 10, 4), dt(2008, 10, 19)),
                       (dt(2008, 11, 25), dt(2008, 12, 17)), (dt(2009, 3, 9), dt(2009, 4, 11)),
                       (dt(2009, 9, 30), dt(2009, 10, 11))]
    #campaign names
    laser_op_name = ['1AB', '2A', '2B', '2C', '3A', '3B', '3C', '3D', '3E', '3F', '3G', '3H', '3I', '3J', '3K', '2D',
                     '2E', '2F']

    # group ICESat dates by operation
    utc_days = ice.UTCTime

    for i in range(len(laser_op_icesat)):
        # datetime to UTC days
        start_dt = datetime_to_yearfraction(laser_op_icesat[i][0]) * 365.2422 - 10
        end_dt = datetime_to_yearfraction(laser_op_icesat[i][1]) * 365.2422 + 10

        mean_dt = 0.5 * (start_dt + end_dt)

        idx = np.logical_and(utc_days < end_dt, utc_days > start_dt)
        utc_days[idx] = mean_dt

    groups = sorted(list(set(list(utc_days))))
    dates = np.array(
        [np.datetime64('01-01-01') + np.timedelta64(int(np.floor(group - 365.2422)), 'D') for group in groups])

    #prepare icesat arrays to pass to wrapper per stack, to avoid reading HDF5/NetCDF multiple times if doing parallel
    icesat_argsin = []
    for fn_stack in list_inters:
        ds = xr.open_dataset(fn_stack)
        tile_name = st.tilename_stack(ds)
        lat, lon = vt.SRTMGL1_naming_to_latlon(tile_name)
        pts_idx = np.logical_and.reduce((ice.lat > lat, ice.lat <= lat + 1, ice.lon > lon, ice.lon <= lon + 1))
        check = np.count_nonzero(pts_idx)
        print('Tile ' + tile_name + ': found ' + str(check) + ' ICESat points')
        if check > 0:
            _, utm = vt.latlon_to_UTM(lat, lon)
            ice.project('epsg:{}'.format(vt.epsg_from_utm(utm)))
            ice_coords = [tup for i, tup in enumerate(ice.xy) if pts_idx[i]]
            ice_elev = ice.elev[pts_idx]
            ice_date = ice.UTCTime[pts_idx]
            ice_latlon = np.array(list(zip(ice.lat[pts_idx],ice.lon[pts_idx])))

            icesat_argsin.append((fn_stack,np.copy(ice_coords),np.copy(ice_latlon),np.copy(ice_elev),np.copy(ice_date),np.copy(groups),np.copy(dates),read_filt,gla_mask,inc_mask,exc_mask))

    if nproc == 1:
        list_h, list_dh, list_zsc, list_dt, list_pos, list_slp, list_time, list_lat, list_lon, list_dh_ref, list_curv, list_dh_tot = ([] for i in range(11))
        for i in range(len(icesat_argsin)):
            tmp_h, tmp_dh, tmp_zsc, tmp_dt, tmp_pos, tmp_slp, tmp_time, tmp_lon, tmp_lat, tmp_dh_ref, tmp_curv, tmp_dh_tot = icesat_comp_wrapper(icesat_argsin[i])
            list_h.append(tmp_h)
            list_dh.append(tmp_dh)
            list_zsc.append(tmp_zsc)
            list_dt.append(tmp_dt)
            list_pos.append(tmp_pos)
            list_slp.append(tmp_slp)
            list_time.append(tmp_time)
            list_lat.append(tmp_lat)
            list_lon.append(tmp_lon)
            list_dh_ref.append(tmp_dh_ref)
            list_curv.append(tmp_curv)
            list_dh_tot.append(tmp_dh_tot)

        h = np.concatenate(list_h)
        dh = np.concatenate(list_dh)
        zsc = np.concatenate(list_zsc)
        dt_out = np.concatenate(list_dt)
        pos = np.concatenate(list_pos)
        slp = np.concatenate(list_slp)
        t = np.concatenate(list_time)
        lat = np.concatenate(list_lat)
        lon = np.concatenate(list_lon)
        dh_ref = np.concatenate(list_dh_ref)
        curv = np.concatenate(list_curv)
        dh_tot = np.concatenate(list_dh_tot)
    else:
        nproc=min(len(icesat_argsin),nproc)
        print('Using '+str(nproc)+' processors...')
        pool = mp.Pool(nproc,maxtasksperchild=1)
        outputs = pool.map(icesat_comp_wrapper,icesat_argsin)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        h = np.concatenate(zip_out[0])
        dh = np.concatenate(zip_out[1])
        zsc = np.concatenate(zip_out[2])
        dt_out = np.concatenate(zip_out[3])
        pos = np.concatenate(zip_out[4])
        slp = np.concatenate(zip_out[5])
        t = np.concatenate(zip_out[6])
        lon = np.concatenate(zip_out[7])
        lat = np.concatenate(zip_out[8])
        dh_ref = np.concatenate(zip_out[9])
        curv = np.concatenate(zip_out[10])
        dh_tot = np.concatenate(zip_out[11])

    return h, dh, zsc, dt_out, pos, slp, t, lon, lat, dh_ref, curv, dh_tot


def comp_stacks_icebridge(list_fn_stack,fn_icebridge,gla_mask=None,inc_mask=None,exc_mask=None,nproc=1,read_filt=False):

    iceb = pd.read_csv(fn_icebridge)

    #get intersecting tiles
    bounds = (iceb.lon.min(), iceb.lon.max(), iceb.lat.min(), iceb.lat.max())
    proj = osr.SpatialReference()
    proj.ImportFromEPSG(4326)
    proj_wkt = proj.ExportToWkt()
    list_inters = get_inters_stack_dem(list_fn_stack,[bounds[0],bounds[2],bounds[1],bounds[3]],proj_wkt)

    if len(list_inters) == 0:
        print('Found no intersection to IceBridge file: '+fn_icebridge)
        raise ValueError('No intersecting file!')


    groups = sorted(list(set(list(pd.to_datetime(iceb.t.values).values.astype('datetime64[D]')))))
    dates = groups

    #prepare icesat arrays to pass to wrapper per stack, to avoid reading HDF5/NetCDF multiple times if doing parallel
    icebridge_argsin = []
    for fn_stack in list_inters:
        ds = xr.open_dataset(fn_stack)
        tile_name = st.tilename_stack(ds)
        lat, lon = vt.SRTMGL1_naming_to_latlon(tile_name)
        pts_idx = np.logical_and.reduce((iceb.lat > lat, iceb.lat <= lat + 1, iceb.lon > lon, iceb.lon <= lon + 1))
        check = np.count_nonzero(pts_idx)
        print('Tile ' + tile_name + ': found ' + str(check) + ' IceBridge points')
        if check > 0:
            _, utm = vt.latlon_to_UTM(lat, lon)
            dest_proj ='epsg:{}'.format(vt.epsg_from_utm(utm))
            dest_proj = pyproj.Proj(init=dest_proj)
            wgs84 = pyproj.Proj(init='epsg:4326')
            x, y = pyproj.transform(wgs84, dest_proj, iceb.lon.values, iceb.lat.values)
            xy = list(zip(x,y))
            ice_coords = [tup for i, tup in enumerate(xy) if pts_idx[i]]
            ice_elev = iceb.h[pts_idx].values
            ice_date = pd.to_datetime(iceb.t[pts_idx].values).values.astype('datetime64[D]')
            ice_latlon = np.array(list(zip(iceb.lat[pts_idx].values,iceb.lon[pts_idx].values)))

            icebridge_argsin.append((fn_stack,np.copy(ice_coords),np.copy(ice_latlon),np.copy(ice_elev),np.copy(ice_date),np.copy(groups),np.copy(dates),read_filt,gla_mask,inc_mask,exc_mask))

    if nproc == 1:
        list_h, list_dh, list_zsc, list_dt, list_pos, list_slp, list_time, list_lat, list_lon, list_dh_ref, list_curv, list_dh_tot = ([] for i in range(11))
        for i in range(len(icebridge_argsin)):
            tmp_h, tmp_dh, tmp_zsc, tmp_dt, tmp_pos, tmp_slp, tmp_time, tmp_lon, tmp_lat, tmp_dh_ref, tmp_curv, tmp_dh_tot = icesat_comp_wrapper(icebridge_argsin[i])
            list_h.append(tmp_h)
            list_dh.append(tmp_dh)
            list_zsc.append(tmp_zsc)
            list_dt.append(tmp_dt)
            list_pos.append(tmp_pos)
            list_slp.append(tmp_slp)
            list_time.append(tmp_time)
            list_lat.append(tmp_lat)
            list_lon.append(tmp_lon)
            list_dh_ref.append(tmp_dh_ref)
            list_curv.append(tmp_curv)
            list_dh_tot.append(tmp_dh_tot)

        h = np.concatenate(list_h)
        dh = np.concatenate(list_dh)
        zsc = np.concatenate(list_zsc)
        dt_out = np.concatenate(list_dt)
        pos = np.concatenate(list_pos)
        slp = np.concatenate(list_slp)
        t = np.concatenate(list_time)
        lat = np.concatenate(list_lat)
        lon = np.concatenate(list_lon)
        dh_ref = np.concatenate(list_dh_ref)
        curv = np.concatenate(list_curv)
        dh_tot = np.concatenate(list_dh_tot)
    else:
        nproc=min(len(icebridge_argsin),nproc)
        print('Using '+str(nproc)+' processors...')
        pool = mp.Pool(nproc,maxtasksperchild=1)
        outputs = pool.map(icesat_comp_wrapper,icebridge_argsin)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        h = np.concatenate(zip_out[0])
        dh = np.concatenate(zip_out[1])
        zsc = np.concatenate(zip_out[2])
        dt_out = np.concatenate(zip_out[3])
        pos = np.concatenate(zip_out[4])
        slp = np.concatenate(zip_out[5])
        t = np.concatenate(zip_out[6])
        lon = np.concatenate(zip_out[7])
        lat = np.concatenate(zip_out[8])
        dh_ref = np.concatenate(zip_out[9])
        curv = np.concatenate(zip_out[10])
        dh_tot = np.concatenate(zip_out[11])

    return h, dh, zsc, dt_out, pos, slp, t, lon, lat, dh_ref, curv, dh_tot

def shift_icesat_stack(fn_ref,fn_icesat,fn_shp):

    _ , _ , shift_params, stats = dem_coregistration(fn_icesat,fn_ref,glaciermask=fn_shp,pts=True,inmem=True)

    return shift_params, stats


def combine_postproc_stacks_tvol(list_fn_stack, fn_shp, feat_id='RGIId', tlim=None, write_combined=True, outdir='.'):
    # get all rgiid intersecting stacks and the list of intersecting stacks
    all_rgiids, list_list_stacks = inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_id)

    # sort by rgiid group with same intersecting stacks
    list_tuples = list(zip(all_rgiids, list_list_stacks))
    grouped = [(k, list(list(zip(*g))[0])) for k, g in groupby(list_tuples, itemgetter(1))]

    # loop through similar combination of stacks (that way, only have to combine/open them once)
    for i in range(len(grouped)):

        list_fn_stack_pack = grouped[i][0]
        rgiid_pack = grouped[i][1]

        list_ds = st.open_datasets(list_fn_stack_pack)
        if len(list_ds) > 1:
            ds = st.combine_stacks(list_ds)
        else:
            ds = list_ds[0]

        if write_combined:
            list_tile = [os.path.splitext(os.path.basename(fn))[0].split('_')[0] for fn in list_fn_stack_pack]
            out_nc = os.path.join(outdir, 'combined_stacks', '_'.join(list_tile))
            ds.to_netcdf(out_nc)

        df_tot = pd.DataFrame()
        # loop through rggiids
        for rgiid in rgiid_pack:
            ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
            layer_name = os.path.splitext(os.path.basename(fn_shp))[0]
            geoimg = st.make_geoimg(ds, 0)
            mask = vt.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg, layer_name=layer_name, feat_id=feat_id, feat_val=rgiid)

            dc, submask, _ = sel_dc(ds, tlim, mask)
            df = int_dc(dc, submask)

            df_tot.append(df)


def get_dt_closest_valid(ds_filt,dates):

    # we read the boolean data cube indicating positions where original data was used

    # first, remove duplicate dates by merging boolean arrays for same dates
    t_vals = list(ds_filt.time.values)
    dates_rm_dupli = sorted(list(set(t_vals)))
    ind_firstdate = []
    for i, date in enumerate(dates_rm_dupli):
        ind_firstdate.append(t_vals.index(date))
    ds_filt2 = ds_filt.isel(time=np.array(ind_firstdate))
    for i in range(len(dates_rm_dupli)):
        t_ind = (t_vals == dates_rm_dupli[i])
        if len(t_ind) > 1:
            ds_filt2.z.values[i, :] = np.any(ds_filt.z[t_vals == dates_rm_dupli[i], :].values.astype(bool), axis=0)

    valid_obs = np.zeros((len(dates),ds_filt2.z.shape[1],ds_filt2.z.shape[2]))
    for i in range(len(dates)-1):
        ind = np.logical_and(dates_rm_dupli>=dates[i],dates_rm_dupli<dates[i+1])
        valid_obs[i,:,:] = np.sum(ds_filt2.z.values[ind,:],axis=0)

    #not robust, works only for monthly (pressé!)
    valid_obs_peryear = np.zeros((len(dates), ds_filt2.z.shape[1], ds_filt2.z.shape[2]))
    for i in range(len(dates) - 1):
        if i % 12 == 0:
            ind = np.logical_and(dates_rm_dupli >= dates[i], dates_rm_dupli < dates[i + 12])
            valid_obs_peryear[i, :, :] = np.any(ds_filt2.z.values[ind, :], axis=0)

    # getting time data as days since 2000
    y0 = np.datetime64('2000-01-01')
    ftime = ds_filt2.time.values
    ftime_delta = np.array([t - y0 for t in ftime])
    days = [td.astype('timedelta64[D]').astype(int) for td in ftime_delta]

    # reindex to get closest not-NaN time value of the date vector
    filt_arr = np.array(ds_filt2.z.values, dtype='float32')
    filt_arr[filt_arr == 0] = np.nan
    days = np.array(days)
    filt_arr = filt_arr * days[:, None, None]
    at_least_2=np.count_nonzero(~np.isnan(filt_arr),axis=0)>=2
    filt_tmp = filt_arr[:,at_least_2]
    out_arr = np.copy(filt_tmp)
    for i in range(np.shape(filt_tmp)[1]):
        ind = ~np.isnan(filt_tmp[:,i])
        fn = interp1d(days[ind],filt_tmp[:,i][ind],kind='nearest',fill_value='extrapolate',assume_sorted=True)
        out_arr[:,i] = fn(days)
    filt_arr[:,at_least_2] = out_arr
    ds_filt2.z.values = filt_arr
    ds_filt_sub = ds_filt2.reindex(time=dates, method='nearest')

    for i in range(len(dates)):
        date = dates[i]
        day_diff = (date - y0).astype('timedelta64[D]').astype(int)
        ds_filt_sub.z.values[i,:] = np.abs(ds_filt_sub.z.values[i, :] - np.ones(ds_filt_sub.z.shape[1:3]) * day_diff)

    return ds_filt_sub, valid_obs, valid_obs_peryear

def sel_int_hypsocheat(argsin):

    # integrate volume without having to know the exact spatial disposition: works for hypsometric (needs only reference elevation)
    # a LOT faster as we don't need to combine (reproject + merge) stacks
    list_fn_stack_pack, shp_params, tlim, i = argsin

    print('Integrating volume for outline group: '+str(i+1))

    list_ds = st.open_datasets(list_fn_stack_pack,load=True)

    dates = list_ds[0].time.values

    list_ds_filt = []
    for fn_stack in list_fn_stack_pack:
        fn_filt = os.path.join(os.path.dirname(fn_stack), os.path.basename(fn_stack).split('_')[0] + '_filtered.nc')
        ds_filt = xr.open_dataset(fn_filt)
        ds_filt.load()
        list_ds_filt.append(ds_filt)

    fn_shp, feat_id, list_feat_val = shp_params
    ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)
    layer_name = os.path.splitext(os.path.basename(fn_shp))[0]

    layer = ds_shp.GetLayer()
    list_lat, list_lon, list_err_area, list_area = ([] for i in range(4))
    for feat_val in list_feat_val:
        for feature in layer:
            if feat_val == feature.GetField(feat_id):
                geom = feature.GetGeometryRef()
                centroid_lon, centroid_lat, _ = geom.Centroid().GetPoint()
                list_lon.append(centroid_lon)
                list_lat.append(centroid_lat)

                #get the area ratio for a 50 meter buffer
                proj_in = osr.SpatialReference()
                proj_in.ImportFromEPSG(4326)
                perc_area_buff, area = vt.get_buffered_area_ratio(geom,proj_in.ExportToWkt(),15.)
                list_area.append(area)
                list_err_area.append(perc_area_buff)
        layer.ResetReading()

    df_tot, df_hyp_tot, df_int_tot = (pd.DataFrame() for i in range(3))
    for feat_val in list_feat_val:
        dh, err, ref, dt, count_area, valid, valid_py = ([] for i in range(7))
        print('Working on feature ID: '+feat_val)
        for i, ds in enumerate(list_ds):

            ds_filt = list_ds_filt[i]

            #get raster equivalent of stack

            geoimg = st.make_geoimg(ds, 0)
            #get mask of latlon tiling
            tile_name = st.tilename_stack(ds)
            mask_tile = vt.latlontile_nodatamask(geoimg,tile_name)
            mask_feat = vt.geoimg_mask_on_feat_shp_ds(ds_shp, geoimg, layer_name=layer_name, feat_id=feat_id, feat_val=feat_val)

            mask = np.logical_and(mask_tile,mask_feat)

            if np.count_nonzero(mask) >0:
                dc, submask, _ = sel_dc(ds,tlim,mask)
                dc_dt, _, _ = sel_dc(ds_filt,tlim,mask)

                dc_dt_sub, valid_obs, valid_obs_peryear = get_dt_closest_valid(dc_dt,dates)

                tmp_dh = dc.z.values[:,submask] - dc.z.values[0,submask]
                tmp_err = np.sqrt(dc.z_ci.values[:,submask]**2 + dc.z_ci.values[0,submask]**2)
                tmp_ref = dc.z.values[0,submask]
                tmp_dt = dc_dt_sub.z.values[:,submask]
                tmp_valid = valid_obs[:, submask]
                tmp_valid_peryear = valid_obs_peryear[:,submask]
                count = np.count_nonzero(mask)

                #exception! to correct in fit_tools at some point?
                #when there is a same-date DEM overlapping twice, and those are the only 2 observations over the whole period, the dh is wrongly of "0" and the dt is only NaNs
                #temporary fix:
                at_least_2 = np.count_nonzero(~np.isnan(tmp_dt), axis=1) >=2
                tmp_dh[~at_least_2, :] = np.nan
                tmp_err[~at_least_2, :] = np.nan

                dh.append(tmp_dh)
                err.append(tmp_err)
                ref.append(tmp_ref)
                dt.append(tmp_dt)
                count_area.append(count)
                valid.append(tmp_valid)
                valid_py.append(tmp_valid_peryear)
            else:
                continue

        ds = list_ds[0]
        x = ds.x.shape[0]
        dx = np.round((ds.x.max().values - ds.x.min().values) / float(x))

        if len(dh)>0 and np.count_nonzero(~np.isnan(np.concatenate(dh,axis=1)))>0:
            dh = np.concatenate(dh,axis=1)
            err = np.concatenate(err,axis=1)
            ref = np.concatenate(ref)
            dt = np.concatenate(dt,axis=1)
            valid = np.concatenate(valid,axis=1)
            valid_py = np.concatenate(valid_py,axis=1)

            df, df_hyp, df_int = volt.hypso_dc(dh,err,ref,dt,dates,np.ones(np.shape(ref),dtype=bool),dx)

            df_int['valid_obs'] = np.nanmean(valid,axis=1)
            df_int['valid_obs_py'] = np.nanmean(valid_py,axis=1)
        else:
            area = np.sum(np.array(count_area))*dx**2
            df = pd.DataFrame()
            df = df.assign(hypso=[np.nan], time=[np.nan], dh=[np.nan], err_dh=[np.nan])
            df_hyp = pd.DataFrame()
            df_hyp = df_hyp.assign(hypso=[np.nan], area_meas=[np.nan], area_tot=[area], nmad=[np.nan])
            df_int = pd.DataFrame()
            df_int = df_int.assign(time=dates, dh=[np.nan]*len(dates), err_dh=[np.nan]*len(dates), perc_area_meas=[np.nan]*len(dates), valid_obs=[np.nan]*len(dates),valid_obs_py=[np.nan]*len(dates))

        df['rgiid'] = feat_val
        df_hyp['rgiid'] = feat_val
        df_int['rgiid'] = feat_val

        df_int['area'] = list_area[list_feat_val.index(feat_val)]
        df_int['lon'] = list_lon[list_feat_val.index(feat_val)]
        df_int['lat'] = list_lat[list_feat_val.index(feat_val)]
        df_int['perc_err_cont'] = list_err_area[list_feat_val.index(feat_val)]

        df_tot = df_tot.append(df)
        df_hyp_tot = df_hyp_tot.append(df_hyp)
        df_int_tot = df_int_tot.append(df_int)

    return df_tot, df_hyp_tot, df_int_tot

def hypsocheat_postproc_stacks_tvol(list_fn_stack, fn_shp, feat_id='RGIId', tlim=None,nproc=64, outfile='int_dh.csv'):

    # get all rgiid intersecting stacks and the list of intersecting stacks
    start = time.time()

    all_rgiids, list_list_stacks = inters_feat_shp_stacks(fn_shp, list_fn_stack, feat_id)

    # sort by rgiid group with same intersecting stacks: !!! commenting to speed up things CPU wise, while using more RAM wise: OK for 100 m stacks
    # all_rgiids = [rgiid for _, rgiid in sorted(zip(list_list_stacks,all_rgiids))]
    # list_list_stacks = sorted(list_list_stacks)

    list_tuples = list(zip(all_rgiids, list_list_stacks))
    grouped = [(k, list(list(zip(*g))[0])) for k, g in groupby(list_tuples, itemgetter(1))]

    print('Found '+str(len(all_rgiids))+' outlines intersecting stacks')
    print('Grouped outlines in '+str(len(grouped))+' packs')

    print('Elapsed: '+str(time.time()-start))
    # loop through similar combination of stacks (that way, only have to combine/open them once)
    df_final, df_hyp_final, df_int_final = (pd.DataFrame() for i in range(3))
    if nproc == 1:

        for i in range(len(grouped)):
            list_fn_stack_pack = grouped[i][0]
            rgiid_pack = grouped[i][1]

            shp_params = (fn_shp,feat_id,rgiid_pack)
            df, df_hyp, df_int = sel_int_hypsocheat((list_fn_stack_pack,shp_params,tlim,i))
            df_final = df_final.append(df)
            df_hyp_final = df_hyp_final.append(df_hyp)
            df_int_final = df_int_final.append(df_int)
    else:
        argsin = [(grouped[i][0], (fn_shp, feat_id, grouped[i][1]), tlim, i) for i in range(len(grouped))]
        pool = mp.Pool(nproc,maxtasksperchild=1)
        outputs = pool.map(sel_int_hypsocheat,argsin,chunksize=1)
        pool.close()
        pool.join()

        print('Finished processing, elapsed: ' + str(time.time() - start))

        zips = list(zip(*outputs))

        dfs = zips[0]
        dfs_hyp = zips[1]
        dfs_int = zips[2]

        print('Finished zipping, elapsed: ' + str(time.time() - start))

        df_final = pd.concat(dfs)
        df_hyp_final = pd.concat(dfs_hyp)
        df_int_final = pd.concat(dfs_int)

        print('Finished putting in dataframes, elapsed: ' + str(time.time() - start))

    fn_csv = os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_hyp2.csv')
    fn_hyp_csv =  os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_hyp.csv')
    fn_int_csv =  os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_int.csv')
    df_final.to_csv(fn_csv,index=False)
    df_hyp_final.to_csv(fn_hyp_csv,index=False)
    df_int_final.to_csv(fn_int_csv,index=False)

def get_base_df_inventory(dir_shp,outfile):

    list_fn_shp = glob(os.path.join(dir_shp,'*/*.shp'),recursive=True)

    df_world = pd.DataFrame()
    for fn_shp in list_fn_shp:

        print('Working on region:'+fn_shp)

        ds_shp = gdal.OpenEx(fn_shp, gdal.OF_VECTOR)

        layer = ds_shp.GetLayer()
        list_rgiid, list_cl, list_nom, list_reg, list_term, list_area, list_lon, list_lat = ([] for i in range(8))
        for feature in layer:
            rgiid = feature.GetField('RGIId')
            geom = feature.GetGeometryRef()

            centroid_lon, centroid_lat, _ = geom.Centroid().GetPoint()

            proj_in = osr.SpatialReference()
            proj_in.ImportFromEPSG(4326)
            _, area = vt.get_buffered_area_ratio(geom, proj_in.ExportToWkt(), 15.)

            print('Glacier: '+rgiid)
            if rgiid[0] != 'G':
                reg = int(rgiid[6:8])
                cl = feature.GetField('Connect')
                nom = feature.GetField('Status')
                term = feature.GetField('TermType')
            else:
                reg = 12
                cl = 0
                nom = 0
                term = 0

            list_rgiid.append(rgiid)
            list_cl.append(cl)
            list_nom.append(nom)
            list_reg.append(reg)
            list_term.append(term)
            list_area.append(area)
            list_lon.append(centroid_lon)
            list_lat.append(centroid_lat)

        df_reg = pd.DataFrame()
        df_reg = df_reg.assign(rgiid=list_rgiid,cl=list_cl,nom=list_nom,reg=list_reg,term=list_term,area=list_area,lon=list_lon,lat=list_lat)

        df_world = df_world.append(df_reg)

    df_world.to_csv(outfile)


def add_base_to_int(df,fn_base,reg):

    # if we can, help ourselves with pre-compiled data for all glaciers
    df_base = pd.read_csv(fn_base)

    # keep only glaciers from the region

    if reg == 20:
        ind = np.logical_or(df_base.reg == 1, df_base.reg == 2)
    elif reg == 21:
        ind = np.logical_or.reduce((df_base.reg == 13, df_base.reg == 14, df_base.reg == 15))
    else:
        ind = df_base.reg == reg
    df_base_reg = df_base[ind]

    # add glaciers completely omitted in this region (no data at all in the tile thus no stack: only concerns 2 tiles in Antarctica?)
    proc_rgiid = list(set(list(df.rgiid)))
    omit_rgiid = [rgiid for rgiid in list(df_base_reg.rgiid) if rgiid not in proc_rgiid]

    print('Added ' + str(len(omit_rgiid)) + ' omitted glaciers.')

    dates = list(set(list(df.time)))
    for rgiid in omit_rgiid:
        df_tmp = pd.DataFrame()
        df_tmp = df_tmp.assign(area=[df_base_reg[df_base_reg.rgiid == rgiid].area.values[0]] * len(dates))
        df_tmp['time'] = dates
        df_tmp['rgiid'] = rgiid
        df_tmp['dh'] = np.nan

        df = df.append(df_tmp)

    # remove nominal glaciers (irrelevant outlines: only North Asia, Scandinavia and Alps)
    ind_nom = df_base_reg.nom == 2
    nom_rgiid = list(df_base_reg.rgiid[ind_nom])

    for rgiid in nom_rgiid:
        df.loc[df.rgiid == rgiid,'dh'] = np.nan

    print('Removed elevation change data for ' + str(len(nom_rgiid)) + ' nominal glaciers.')

    # remove CL2 glaciers (Greenland)
    ind_cl2 = df_base_reg.cl == 2
    cl2_rgiid = list(df_base_reg.rgiid[ind_cl2])

    for rgiid in cl2_rgiid:
        df.loc[df.rgiid == rgiid,'area'] = np.nan

    print('Removed contributing area for ' + str(len(cl2_rgiid)) + ' CL2 glaciers.')

    return df

def aggregate_int_to_all(df,nproc=1,get_corr_err=True):

    # get area of glacier without any data to account for it later
    valid = np.logical_and.reduce((~np.isnan(df.dh), ~np.isnan(df.area),~np.isnan(df.err_dh)))
    df_nodata = df[~valid]
    rgiid_nodata = list(set(list(df_nodata.rgiid)))
    area_nodata = 0
    for rgiid in rgiid_nodata:
        area_missed = df_nodata[df_nodata.rgiid == rgiid].area.values[0]
        if ~np.isnan(area_missed):
            area_nodata += area_missed

    # keep only valid glaciers
    print('Removed '+str(len(rgiid_nodata))+' non-valid glaciers with total area of '+str(area_nodata/1000000.)+' km².')
    print(rgiid_nodata)
    reg = df.reg.values[0]
    df = df[valid]

    # get the regional volume
    df['dvol'] = df['dh'] * df['area']
    df['var_cont'] = df['perc_err_cont'] * df['area']
    df['area_res'] = df['area'] * df['perc_area_res']
    df['area_meas'] = df['area'] * df['perc_area_meas']
    df['area_valid_obs'] = df['area'] * df['valid_obs']
    df['area_valid_obs_py'] = df['area'] * df['valid_obs_py']
    df_tot = df.groupby('time')['dvol', 'area', 'var_cont', 'area_meas', 'area_res', 'area_valid_obs','area_valid_obs_py'].sum()

    # propagate volume change accouting for area of no data glaciers
    df_tot['dvol'] *= (df_tot['area'] + area_nodata) / df_tot['area']
    df_tot['perc_err_cont'] = df_tot['var_cont'] / df_tot['area']
    df_tot['valid_obs'] = df_tot['area_valid_obs'] / df_tot['area']
    df_tot['valid_obs_py'] = df_tot['area_valid_obs_py'] / df_tot['area']
    df_tot['area'] += area_nodata
    df_tot['perc_area_meas'] = df_tot['area_meas'] / df_tot['area']
    df_tot['perc_area_res'] = df_tot['area_res'] / df_tot['area']
    df_tot['dh'] = df_tot['dvol'] / df_tot['area']
    df_tot['area_nodata'] = area_nodata
    df_tot['reg'] = reg

    df_tot = df_tot.drop(columns=['var_cont', 'area_meas', 'area_res', 'area_valid_obs'])

    # integrate elevation change error accouting for spatial correlation of temporal interpolation between glaciers

    if np.count_nonzero(valid) == 0:
        return df_tot

        # get error only for annual
    tlim = [np.datetime64(str(2000 + i) + '-01-01') for i in range(21)]
    times = sorted(list(set(list(df['time']))))
    df_time = pd.DataFrame()
    df_time = df_time.assign(time=times)
    df_time.index = pd.DatetimeIndex(pd.to_datetime(times))
    closest_dates = []
    for tl in tlim:
        time_clos = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tl), method='nearest')][0]
        closest_dates.append(time_clos)
    # int_err = []
    df_tot['err_dh'] = np.nan
    if get_corr_err:
        for time in closest_dates:
            print('Time step: ' + time)
            df_tmp = df[df.time == time]

            corr_ranges = [150, 2000, 5000, 20000, 50000, 200000]
            list_tuple_errs = list(
                zip(*[df_tmp['err_corr_' + str(corr_ranges[i])].values for i in range(len(corr_ranges))]))
            list_area_tot = df_tmp.area.values
            list_lat = df_tmp.lat.values
            list_lon = df_tmp.lon.values
            err = spt.double_sum_covar(list_tuple_errs, corr_ranges, list_area_tot, list_lat, list_lon, nproc=nproc)
            df_tot.loc[df_tot.index == time,'err_dh'] = err

    # int_err = np.array(int_err)
    # df_tot['err_dh'] = int_err

    # error on general inventory, uncharted glaciers ; 3 percent here
    # df_tot['err_area'] = 3./100.*df_tot['area']

    # propagate error to volume change
    # df_tot['err_dvol'] = np.sqrt((df_tot['err_dh']*df_tot['area'])**2 + (df_tot['dh']*np.sqrt(df_tot['err_area']**2+df_tot['err_cont']**2))**2)
    df_tot['err_dvol'] = np.sqrt((df_tot['err_dh'] * df_tot['area']) ** 2 + (
                df_tot['dh'] * df_tot['perc_err_cont'] / 100. * df_tot['area']) ** 2)

    # convert to mass change (Huss, 2013)
    df_tot['dm'] = df_tot['dvol'] * 0.85 / 10 ** 9

    # propagate error to mass change (Huss, 2013)
    df_tot['err_dm'] = np.sqrt((df_tot['err_dvol'] * 0.85 / 10 ** 9) ** 2 + (
            df_tot['dvol'] * 0.06 / 10 ** 9) ** 2)

    return df_tot

def df_int_to_base(infile,fn_base=None):

    df_init = pd.read_csv(infile)

    print('Working on ' + infile)

    region = int(os.path.basename(infile).split('_')[1])
    # to integrate properly regions 1, 2, 13, 14, 15
    if region == 1:
        list_reg = [20, 1, 2]
    elif region == 13:
        list_reg = [21, 13, 14, 15]
    else:
        list_reg = [region]

    for reg in list_reg:

        print('Processing for region ' + str(reg))

        if reg > 19:
            fn_base_out = os.path.join(os.path.dirname(infile),
                                       os.path.splitext(os.path.basename(infile))[0] + '_base.csv')
            df = df_init
        else:
            fn_base_out = os.path.join(os.path.dirname(infile), 'dh_' + str(reg).zfill(2) + '_rgi60_int_base.csv')
            if reg in [1, 2, 13, 14, 15]:
                ind = np.array(['RGI60-' + str(reg).zfill(2) in rgiid for rgiid in list(df_init.rgiid.values)])
                df = df_init[ind]
            else:
                df = df_init

        df['reg'] = reg

        if fn_base is not None:
            df = add_base_to_int(df, fn_base, reg)
            # if not os.path.exists(fn_base_out):
            df.to_csv(fn_base_out)

def df_int_to_reg(infile,nproc=1):

    df = pd.read_csv(infile)

    fn_reg_out = os.path.join(os.path.dirname(infile),
                              os.path.splitext(os.path.basename(infile))[0] + '_reg.csv')

    #let's go
    df_tot = aggregate_int_to_all(df,nproc=nproc)
    df_tot.to_csv(fn_reg_out)


def aggregate_all_to_multiannual(df,mult_ann=1,fn_tarea=None,frac_area=None):

    if np.count_nonzero(~np.isnan(df.dh))==0:
        df_mult_ann = pd.DataFrame()
        # df_mult_ann['reg'] = df.reg.values[0]
        # df_mult_ann['perc_area_meas'] = df.perc_area_meas.values[0]
        # df_mult_ann['perc_area_res'] = df.perc_area_res.values[0]
        # df_mult_ann['area_nodata'] = df.area_nodata.values[0]
        # df_mult_ann['area'] = df.area.values[0]
        return df_mult_ann

    # find closest annual dates to monthly time series
    nb_period = int(np.floor(20 / mult_ann))
    tlim = [np.datetime64(str(2000 + mult_ann * i) + '-01-01') for i in range(nb_period + 1)]
    times = sorted(list(set(list(df['time']))))
    df_time = pd.DataFrame()
    df_time = df_time.assign(time=times)
    df_time.index = pd.DatetimeIndex(pd.to_datetime(times))
    closest_dates = []
    for tl in tlim:
        time_clos = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tl), method='nearest')][0]
        closest_dates.append(time_clos)

    reg = df.reg.values[0]

    if fn_tarea is not None:
        df_tarea = pd.read_csv(fn_tarea)

        # get regionally evolving areas
        if reg == 20:
            tmp_tarea = df_tarea['RGI1'].values + df_tarea['RGI2'].values
        elif reg == 21:
            tmp_tarea = df_tarea['RGI13'].values + df_tarea['RGI14'].values + df_tarea['RGI15'].values
        else:
            tmp_tarea = df_tarea['RGI' + str(int(reg))].values

        if frac_area is not None:
            tmp_tarea = frac_area*tmp_tarea

        tarea = np.zeros(len(tlim))
        for i in range(nb_period + 1):
            # getting years 2000 to 2020
            ind = df_tarea['YEAR'] == 2000 + i * mult_ann
            tarea[i] = tmp_tarea[ind][0] * 1000000
    else:
        tarea = np.repeat(df.area.values[0], len(tlim))

    # derive dh rates
    list_tarea, list_dhdt, list_err_dhdt, list_dvoldt, list_err_dvoldt, list_dmdt, list_err_dmdt, list_valid_obs, list_dt, list_valid_obs_py = ([] for i in range(10))
    for i in range(len(tlim) - 1):
        # derive volume change for subperiod
        area = df.area.values[0]
        dvol = (df[df.time == closest_dates[i + 1]].dvol.values - df[df.time == closest_dates[i]].dvol.values)[0]
        dh = dvol / area

        err_dh = np.sqrt(
            df[df.time == closest_dates[i + 1]].err_dh.values[0] ** 2 + df[df.time == closest_dates[i]].err_dh.values[
                0] ** 2)
        err_dvol = np.sqrt((err_dh * area) ** 2 + (dh * df.perc_err_cont.values[0] / 100. * area) ** 2)

        dvoldt = dvol / mult_ann
        err_dvoldt = err_dvol / mult_ann

        dmdt = dvol * 0.85 / 10 ** 9 / mult_ann
        err_dmdt = np.sqrt((err_dvol * 0.85 / 10 ** 9) ** 2 + (
                dvol * 0.06 / 10 ** 9) ** 2) / mult_ann

        linear_area = (tarea[i] + tarea[i + 1]) / 2
        dhdt = dvol / linear_area / mult_ann
        perc_err_linear_area = 1. / 100
        err_dhdt = np.sqrt((err_dvol / linear_area) ** 2 \
                           + (perc_err_linear_area * linear_area * dvol / linear_area ** 2) ** 2) / mult_ann

        ind = np.logical_and(df.time >= closest_dates[i], df.time < closest_dates[i + 1])
        valid_obs = np.nansum(df.valid_obs[ind].values)
        valid_obs_py = np.nansum(df.valid_obs_py[ind].values)

        list_tarea.append(linear_area)
        list_dhdt.append(dhdt)
        list_err_dhdt.append(err_dhdt)
        list_dvoldt.append(dvoldt)
        list_err_dvoldt.append(err_dvoldt)
        list_dmdt.append(dmdt)
        list_err_dmdt.append(err_dmdt)
        list_dt.append(str(tlim[i]) + '_' + str(tlim[i + 1]))
        list_valid_obs.append(valid_obs)
        list_valid_obs_py.append(valid_obs_py)

    df_mult_ann = pd.DataFrame()
    df_mult_ann = df_mult_ann.assign(period=list_dt, tarea=list_tarea, dhdt=list_dhdt, err_dhdt=list_err_dhdt,
                                     dvoldt=list_dvoldt, err_dvoldt=list_err_dvoldt, dmdt=list_dmdt,
                                     err_dmdt=list_err_dmdt, valid_obs=list_valid_obs, valid_obs_py =list_valid_obs_py)
    df_mult_ann['reg'] = reg
    df_mult_ann['perc_area_meas'] = df.perc_area_meas.values[0]
    df_mult_ann['perc_area_res'] = df.perc_area_res.values[0]
    df_mult_ann['area_nodata'] = df.area_nodata.values[0]
    df_mult_ann['area'] = df.area.values[0]

    return df_mult_ann


def df_region_to_periods(infile_reg,fn_tarea=None,frac_area=None):

    df = pd.read_csv(infile_reg)
    fn_out = os.path.join(os.path.dirname(infile_reg),
                          os.path.splitext(os.path.basename(infile_reg))[0] + '_subperiods.csv')

    list_df = []
    for mult_ann in [1,2,4,5,10,20]:
        df_mult_ann = aggregate_all_to_multiannual(df,mult_ann=mult_ann,fn_tarea=fn_tarea,frac_area=frac_area)
        list_df.append(df_mult_ann)

    df_final = pd.concat(list_df)
    df_final.to_csv(fn_out)


def wrapper_tile_rgiids(argsin):

    list_tile, df_base, uniq_rgiids, i, itot = argsin

    print('Working on pack: '+str(i)+' out of '+str(itot))

    tmp_rgiids = []
    for j, tile in enumerate(list_tile):

        print('Tile: '+str(j)+' out of '+str(len(list_tile)))

        ind_base = np.logical_and.reduce((df_base.lat >= tile[1], df_base.lat < tile[3], df_base.lon >= tile[0], df_base.lon < tile[2]))
        rgiids_base = list(df_base[ind_base].rgiid)
        if len(rgiids_base)==0:
            tmp_rgiids.append(['NA'])
            print('No base glacier in this tile, skipping...')
            continue

        #this is only necessary is the few void glaciers have not been added to the dataframe post volume integ
        rgiids = [rgiid for rgiid in rgiids_base if rgiid in uniq_rgiids]
        if len(rgiids) == 0:
            tmp_rgiids.append(['NA'])
            print('No glacier in this tile, skipping...')
            continue

        tmp_rgiids.append(rgiids)

    return tmp_rgiids

def wrapper_tile_int_to_all_to_mult_ann(argsin):

    df_tile, tile, i, imax = argsin

    # df_tile = df_keep[df_keep.rgiid.isin(rgiids_final)]

    print('Working on tile '+str(tile)+': '+str(i)+' out of '+str(imax))

    df_agg = aggregate_int_to_all(df_tile, nproc=1)
    df_agg['time'] = df_agg.index.values

    list_df_mult = []
    for mult_ann in [1, 2, 4, 5, 10, 20]:
        df_mult = aggregate_all_to_multiannual(df_agg, mult_ann=mult_ann)
        list_df_mult.append(df_mult)

    df_mult_all = pd.concat(list_df_mult)
    df_mult_all['tile_lonmin'] = tile[0]
    df_mult_all['tile_latmin'] = tile[1]

    return df_mult_all

def df_all_base_to_tile(list_fn_int_base,base_df,tile_size=1,nproc=1,sort_tw=True):

    fn_out = os.path.join(os.path.dirname(list_fn_int_base[0]), 'dh_world_tiles_' + str(tile_size) + 'deg.csv')

    def world_latlon_tiles(deg_size):

        list_tile = []
        for ymin in np.arange(-90, 90, deg_size):
            for xmin in np.arange(-180, 180, deg_size):
                list_tile.append(np.array([xmin, ymin, xmin + deg_size, ymin + deg_size], dtype=float))

        return list_tile

    list_tile = world_latlon_tiles(tile_size)
    # list_tile = [np.array([-180,70,-179,71],dtype=float),np.array([-18,64,-17,65],dtype=float),np.array([-17,64,-16,65],dtype=float)]

    df_base = pd.read_csv(base_df)

    if sort_tw:
        ind_tw = df_base.term == 1
        keep_tw = list(df_base.rgiid[ind_tw])
        keep_ntw = list(df_base.rgiid[~ind_tw])
        list_keeps = [keep_tw,keep_ntw,list(df_base.rgiid)]
        name_keeps = ['tw','ntw','all']
    else:
        list_keeps=[list(df_base.rgiid)]
        name_keeps=['all']

    list_df_final = []
    # list_dfs = []
    for fn_int_base in list_fn_int_base:
        df = pd.read_csv(fn_int_base)
    #     list_dfs.append(pd.read_csv(fn_int_base))
    # df = pd.concat(list_dfs)

        for keep in list_keeps:

            print('Processing for category: '+name_keeps[list_keeps.index(keep)])

            df_keep=df[df.rgiid.isin(keep)]

            if len(df_keep)==0:
                continue

            r = int(os.path.basename(fn_int_base).split('_')[1])
            if r == 1:
                list_reg = [1,2]
            elif r == 13:
                list_reg = [13,14,15]
            else:
                list_reg = [r]

            list_ext_reg = []
            # list_reg = list(set(list(df_base.reg)))
            print('Deriving region extents for tile omission...')
            for reg in list_reg:
                print('Region: '+str(reg))
                df_tmp = df_base[df_base.reg==reg]
                list_ext_reg.append(np.array([df_tmp.lon.min(),df_tmp.lat.min(),df_tmp.lon.max(),df_tmp.lat.max()],dtype=float))

            uniq_rgiids = list(set(list(df_keep.rgiid)))

            print('Removing useless tiles')
            list_tile_possib = []
            for tile in list_tile:
                # to speed things up, let's remove all tiles outside region extents
                poly_tile = vt.poly_from_extent(tile)
                chk = 0
                for reg in list_reg:
                    poly = vt.poly_from_extent(list_ext_reg[list_reg.index(reg)])
                    if poly.Intersects(poly_tile):
                        chk = 1
                if chk != 0:
                    list_tile_possib.append(tile)

            print('Finding which tiles have to be processed...')
            if nproc==1:
                print('Using 1 core...')
                list_rgiids = []
                for i in range(len(list_tile_possib)):
                    tmp_rgiids = wrapper_tile_rgiids((list_tile_possib[i], df_base, uniq_rgiids, i, len(list_tile_possib)))
                    list_rgiids.append(tmp_rgiids)
            else:
                print('Using '+str(nproc)+' cores...')
                pack_size = int(np.ceil(len(list_tile_possib) / nproc))
                argsin = [(list_tile_possib[i:min(i+pack_size,len(list_tile_possib))], df_base, uniq_rgiids, k, nproc) for k ,i in enumerate(np.arange(0,len(list_tile_possib),pack_size))]
                pool = mp.Pool(nproc, maxtasksperchild=1)
                outputs = pool.map(wrapper_tile_rgiids, argsin, chunksize=1)
                pool.close()
                pool.join()

                list_rgiids = list(itertools.chain(*outputs))


            list_df_tile = []
            list_tile_final = []
            # list_rgiids_final = []
            for i, rgiids in enumerate(list_rgiids):
                if len(rgiids)==1 and rgiids[0]=='NA':
                    continue
                # list_rgiids_final.append(rgiids)
                df_tile = df_keep[df_keep.rgiid.isin(rgiids)]
                list_df_tile.append(df_tile)
                list_tile_final.append(list_tile_possib[i])
            print('Found '+str(len(list_tile_final))+' valid tiles to integrate.')

            print('Integrating...')
            if nproc == 1:
                print('Using 1 core...')
                list_mult_all = []
                for i, tile_final in enumerate(list_tile_final):
                    df_mult_all = wrapper_tile_int_to_all_to_mult_ann((list_df_tile[i],tile_final,i,len(list_tile_final)))
                    list_mult_all.append(df_mult_all)
            else:
                print('Using '+str(nproc)+' cores...')
                argsin = [(list_df_tile[i],list_tile_final[i],i,len(list_tile_final)) for i in range(len(list_tile_final))]
                pool = mp.Pool(nproc, maxtasksperchild=1)
                outputs = pool.map(wrapper_tile_int_to_all_to_mult_ann, argsin, chunksize=1)
                pool.close()
                pool.join()

                list_mult_all = outputs

            if len(list_mult_all)>0:
                df_tiles = pd.concat(list_mult_all)
                df_tiles['tile_size'] = tile_size
                df_tiles['category'] = name_keeps[list_keeps.index(keep)]
                df_tiles['reg'] = os.path.basename(fn_int_base).split('_')[1]
                list_df_final.append(df_tiles)

    df_cat = pd.concat(list_df_final)
    df_cat.to_csv(fn_out)


def aggregate_df_int_time(infile,tlim=None,rate=False):

    df = pd.read_csv(infile)

    #not the cleanest closest time search you can write, but works
    times = sorted(list(set(list(df['time']))))
    df_time = pd.DataFrame()
    df_time = df_time.assign(time=times)
    df_time.index = pd.DatetimeIndex(pd.to_datetime(times))
    time_start = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tlim[0]),method='nearest')][0]
    time_end = df_time.iloc[df_time.index.get_loc(pd.to_datetime(tlim[1]),method='nearest')][0]

    df = df.sort_values(by='rgiid')
    df_start = df[df.time == time_start]
    df_end = df[df.time == time_end]

    ind = np.logical_and(df.time >= time_start, df.time < time_end)
    df_period = df[ind]
    df_period_cum = df_period.groupby('rgiid')['valid_obs',].sum()

    df_gla = pd.DataFrame()
    df_gla['rgiid'] = df_start['rgiid']
    df_gla['valid_obs'] = df_period_cum.valid_obs.values
    df_gla['period'] = str(time_start)+'_'+str(time_end)
    df_gla['area'] = df_start['area']
    df_gla['perc_area_meas'] = df_start['perc_area_meas']
    df_gla['lat'] = df_start['lat']
    df_gla['lon'] = df_start['lon']
    df_gla['dh'] = df_end['dh'].values - df_start['dh'].values
    df_gla['err_dh'] = np.sqrt(df_end['err_dh'].values**2+df_start['err_dh'].values**2)
    df_gla['err_cont'] = df_start['perc_err_cont'] * df_start['area'] / 100.
    df_gla['perc_err_cont']=df_start['perc_err_cont']

    #get the volume per glacier
    df_gla['dvol'] = df_gla['dh'] * df_gla['area']

    #correct systematic seasonal biases (snow-covered terrain)
    # df_tot['dh'] =

    #propagate error to volume change
    df_gla['err_dvol'] = np.sqrt((df_gla['err_dh'] * df_gla['area']) ** 2 + (
            df_gla['dh'] * df_gla['perc_err_cont'] / 100. * df_gla['area']) ** 2)

    #convert to mass change (Huss, 2013)
    df_gla['dm'] = df_gla['dvol'] * 0.85 / 10 ** 9

    #propagate error to mass change (Huss, 2013)
    df_gla['err_dm'] = np.sqrt((df_gla['err_dvol'] * 0.85 / 10 ** 9) ** 2 + (
                df_gla['dvol'] * 0.06 / 10 ** 9) ** 2)

    df_gla['dmda'] = df_gla['dm'].values / df_gla['area'].values * 10 ** 9
    df_gla['err_dmda'] = np.sqrt((df_gla['err_dm'].values * 10 ** 9 / df_gla['area'].values) ** 2 \
                         + (df_gla['err_cont'].values * df_gla['dm'].values * 10 ** 9 / df_gla['area'].values ** 2) ** 2)

    if rate:
        dt = (tlim[1]-tlim[0]).astype(int)/365.2524
        df_gla['dmdtda'] = df_gla['dmda'] / dt
        df_gla['err_dmdtda'] = df_gla['err_dmda'] /dt
        df_gla['dhdt'] = df_gla['dh']/dt
        df_gla['err_dhdt'] = df_gla['err_dh'] /dt

        df_gla = df_gla.drop(columns=['dvol', 'err_dvol', 'dm', 'err_dm','dh','err_dh','dmda','err_dmda'])
    else:
        df_gla = df_gla.drop(columns=['dvol', 'err_dvol', 'dm', 'err_dm','dh','err_dh'])

    df_gla['area'] = df_gla['area']/1000000
    df_gla['reg'] = df_start['reg']

    return df_gla

