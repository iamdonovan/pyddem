"""
pyddem.fit_tools provides tools to derive filter and interpolate DEM stacks into elevation time series
"""
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import numpy as np
import gdal
from dask.diagnostics import ProgressBar
import pandas as pd
import functools
import xarray as xr
import geopandas
import matplotlib.pylab as plt
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import chain
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import filters
from skimage.morphology import disk
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, ExpSineSquared as ESS, \
    PairwiseKernel
from numba import jit, vectorize, float32
from llc import jit_filter_function
from pybob.GeoImg import GeoImg
from pybob.coreg_tools import get_slope
from pybob.image_tools import create_mask_from_shapefile
from pybob.plot_tools import set_pretty_fonts
from pybob.bob_tools import mkdir_p
from pybob.plot_tools import plot_polygon_df
import pyddem.stack_tools as st
import pyddem.tdem_tools as tt
import pyddem.vector_tools as vt
from pybob.ddem_tools import nmad
from warnings import filterwarnings

filterwarnings('ignore')


def make_dh_animation(ds, fn_shp=None,month_a_year=None, rates=False, figsize=(10,10), t0=None, t1=None, dh_max=20, var='z', cmap='RdYlBu',
                      xlbl='easting (km)',
                      ylbl='northing (km)'):
    """
       Generate a GIF from elevation time series

       :param ds: xarray Dataset of elevation time series
       :param month_a_year: Numerical month to keep only one month per year in the animation
       :param rates: Display rates instead of cumulative change
       :param figsize: Tuple of figure size
       :param t0: Starting date
       :param t1: End date
       :param dh_max: Max scale for colorbar
       :param var: Variable to display in the dataset
       :param cmap: Colormap
       :param xlbl: xlabel for animation
       :param ylbl: ylabel for animation

       :returns: Figure, List of images and annotations for animation
       """

    set_pretty_fonts()
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    ds_sub = ds.sel(time=slice(t0, t1))

    if month_a_year is not None:
        t_vals = ds_sub.time.values
        y0 = t_vals[0].astype('datetime64[D]').astype(object).year
        y1 = t_vals[-1].astype('datetime64[D]').astype(object).year

        t_vec = []
        for y in np.arange(y0, y1, 1):
            t = np.datetime64(str(y) + '-' + str(month_a_year).zfill(2) + '-01')
            td = np.array([(t - t_vals[i].astype('datetime64[D]')).astype(int) for i in range(len(t_vals))])

            closer_dat = t_vals[(np.abs(td) == np.min(np.abs(td)))][0]
            t_vec.append(closer_dat)

        ds_sub = ds.sel(time=t_vec)
        # mid = int(np.floor(len(ds_sub.time.values)/2))

    if var == 'z':
        if rates:
            dh_ = np.diff(ds_sub.variables[var].values,axis=0)
        else:
            dh_ = ds_sub.variables[var].values - ds_sub.variables[var].values[0]
    elif var == 'z_ci':
        dh_ = ds_sub.variables[var].values

    times = np.array([np.datetime_as_string(t.astype('datetime64[D]')) for t in ds_sub.time.values])
    nice_ext = np.array([ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()]) / 1000
    ims = []

    im = ax.imshow(dh_[0], extent=nice_ext, vmin=-dh_max, vmax=dh_max, cmap=cmap)
    if rates:
        annot = times[0]+'-'+times[1]
    else:
        annot = times[0]
    ann = ax.annotate(annot, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=20,
                      fontweight='bold', color='black', family='monospace')
    ims.append([im, ann])

    tmp_geoimg = st.make_geoimg(ds)
    df = geopandas.read_file(fn_shp).to_crs({'init':'epsg:'+str(tmp_geoimg.epsg)})
    fig, _ = plot_polygon_df(df,fig=fig,ax=ax,facecolor='None',edgecolor='black',zorder=30)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, extend='both')


    cax.set_ylabel('elevation change (m)')
    ax.set_ylabel(ylbl)
    ax.set_xlabel(xlbl)

    if rates:
        l = len(times[1:])-1
    else:
        l = len(times[1:])
    for i in range(l):
        im = ax.imshow(dh_[i + 1], vmin=-dh_max, vmax=dh_max, cmap=cmap, extent=nice_ext)
        if rates:
            annot = times[i+1] + '-' + times[i+2]
        else:
            annot = times[i+1]
        ann = ax.annotate(annot, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=20,
                          fontweight='bold', color='black', family='monospace')
        ims.append([im, ann])

    plt.tight_layout()

    return fig, ims


def write_animation(fig, ims, outfilename='output.gif', ani_writer='imagemagick', **kwargs):
    ani = animation.ArtistAnimation(fig, ims, **kwargs)
    ani.save(outfilename, writer=ani_writer)


def get_dem_date(ds, ds_filt, t, outname):
    """
       Extract a DEM from an elevation time series, with elevation error and time lag to the closest observation

       :param ds: xarray Dataset of elevation time series
       :param ds_filt: xarray dataset of boolean data cube of valid observation used to generate the time series
       ;param t: Date of extraction
       :outname: Filename for output rasters

       :returns:
       """

    tmp_img = st.make_geoimg(ds)

    dc = ds.interp(time=[t])

    h = dc.variables['z'].values[0]
    err = dc.variables['z_ci'].values[0]

    dates = [t]
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

    for i in range(len(dates)):
        date = dates[i]
        day_diff = (date - y0).astype('timedelta64[D]').astype(int)
        ds_filt_sub.z.values[i, :] = np.abs(ds_filt_sub.z.values[i, :] - np.ones(ds_filt_sub.z.shape[1:3]) * day_diff)

    dt = ds_filt_sub.z.values[0]

    tmp_img.img = h
    tmp_img.write(os.path.join(os.path.dirname(outname), os.path.basename(outname) + '_AST_SRTM_h.tif'))
    tmp_img.img = err
    tmp_img.write(os.path.join(os.path.dirname(outname), os.path.basename(outname) + '_AST_SRTM_herr.tif'))
    tmp_img.img = dt
    tmp_img.write(os.path.join(os.path.dirname(outname), os.path.basename(outname) + '_AST_SRTM_dt.tif'))


def get_dem_date_exact(ds, t, outname):
    tmp_img = st.make_geoimg(ds)

    times = sorted(list(set(list(ds.time.values))))
    df_time = pd.DataFrame()
    df_time = df_time.assign(time=times)
    df_time.index = pd.DatetimeIndex(pd.to_datetime(times))
    t_exact = df_time.iloc[df_time.index.get_loc(pd.to_datetime(t), method='nearest')][0]

    dc = ds.sel(time=[t_exact])

    h = dc.variables['z'].values[0]

    tmp_img.img = h
    tmp_img.write(os.path.join(os.path.dirname(outname), os.path.basename(outname) + '_ASTER_recons_h.tif'))


def get_full_dh(ds, outname, t0=None, t1=None):
    tmp_img = st.make_geoimg(ds)

    #filter pixel with 2000-2019 error superior than 500 m (very large, because it is overestimated in many instances)
    err_tot = np.sqrt(ds.variables['z_ci'].values[-1] ** 2 + ds.variables['z_ci'].values[0] ** 2)
    ind = err_tot > 500.

    dc = ds.sel(time=slice(t0, t1))

    dh = dc.variables['z'].values[-1] - dc.variables['z'].values[0]
    err = np.sqrt(dc.variables['z_ci'].values[-1] ** 2 + dc.variables['z_ci'].values[0] ** 2)

    dh[ind] = np.nan
    err[ind] = np.nan

    mkdir_p(os.path.join(os.path.dirname(outname), 'dh'))
    mkdir_p(os.path.join(os.path.dirname(outname), 'dh_err'))

    period = str(t0) + '_' + str(t1)
    tmp_img.img = dh
    tmp_img.write(os.path.join(os.path.dirname(outname), 'dh', os.path.basename(outname) + '_'+period+'_dh.tif'))
    tmp_img.img = err
    tmp_img.write(os.path.join(os.path.dirname(outname), 'dh_err',os.path.basename(outname) + '_'+period+'_dh_err.tif'))

def reproj_build_vrt(list_dh, utm, out_vrt, res):
    epsg = vt.epsg_from_utm(utm)

    list_fn_out = []
    for dh in list_dh:

        tile_name = os.path.basename(dh).split('_')[0]
        print(tile_name)
        img = GeoImg(dh)
        dest = gdal.Warp('', img.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg), resampleAlg=gdal.GRA_NearestNeighbour,
                         xRes=res, yRes=res)

        try:
            out_img = GeoImg(dest)
        except:
            print('Could not reproject in that projection.')
            continue
        nodata_mask = vt.latlontile_nodatamask(out_img, tile_name)
        out_img.img[~nodata_mask] = np.nan

        fn_out = os.path.join(os.path.dirname(dh), os.path.splitext(os.path.basename(dh))[0] + '_' + str(epsg) + '.tif')
        list_fn_out.append(fn_out)
        out_img.write(fn_out)

    gdal.BuildVRT(out_vrt, list_fn_out, resampleAlg='bilinear')


def get_stack_mask(maskshp, ds):
    npix_y, npix_x = ds['z'][0].shape
    dx = np.round((ds.x.max().values - ds.x.min().values) / float(npix_x))
    dy = np.round((ds.y.min().values - ds.y.max().values) / float(npix_y))

    newgt = (ds.x.min().values - 0, dx, 0, ds.y.max().values - 0, 0, dy)

    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)

    sp = dst.SetProjection(ds.crs.spatial_ref)
    sg = dst.SetGeoTransform(newgt)

    wa = dst.GetRasterBand(1).WriteArray(ds['z'][0].values)
    md = dst.SetMetadata({'Area_or_point': 'Point'})
    del wa, sg, sp, md

    img = GeoImg(dst)
    mask = create_mask_from_shapefile(img, maskshp)
    return mask


def vgm_1d(argsin):
    """
    1D variogram sampling

    :param argsin: tuple of x values, y values, lag cutoff value and interval of x sampling (for easier parallel proc)

    :returns: semivariance
    """

    t_vals, detrend_elev, lag_cutoff, tstep = argsin

    # 1D variogram: inspired by http://connor-johnson.com/2014/03/20/simple-kriging-in-python/

    def SVh(P, h, bw):
        # empirical variogram for a single lag
        pd = np.abs(np.subtract.outer(P[:, 0], P[:, 0]))
        N = pd.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i + 1, N):
                if (pd[i, j] >= h - bw) and (pd[i, j] <= h + bw):
                    Z.append((P[i, 1] - P[j, 1]) ** 2.0)
        if len(Z) > 0:
            return np.nansum(Z) / (2.0 * len(Z)), len(Z)
        else:
            return np.nan, 0

    def SV(P, hs, bw):
        # empirical variogram for a collection of lags
        sv = list()
        p = list()
        for h in hs:
            svh, ph = SVh(P, h, bw)
            sv.append(svh)
            p.append(ph)
        sv = [[sv[i], p[i]] for i in range(len(hs))]
        return np.array(sv).T

    ind_valid = ~np.isnan(detrend_elev)
    sample = np.column_stack((t_vals[ind_valid], detrend_elev[ind_valid]))

    hs = np.arange(0, lag_cutoff, tstep)
    sv = SV(sample, hs, tstep)

    return sv


def wrapper_vgm1d(argsin):
    t_vals, stack_detrend_elev, k, lag_cutoff, tstep = argsin

    _, nb_iter = np.shape(stack_detrend_elev)

    print('Pack of ' + str(nb_iter) + ' variograms number ' + str(k))

    lags = np.arange(0, lag_cutoff, tstep)

    vdata = np.zeros((len(lags), nb_iter))
    pdata = np.zeros((len(lags), nb_iter))
    for i in np.arange(nb_iter):
        sv = vgm_1d((t_vals, stack_detrend_elev[:, i], lag_cutoff, tstep))
        vdata[:, i] = sv[0]
        pdata[:, i] = sv[1]

    return vdata, pdata


def vgm_1d_med(argsin):
    """
    1D median variogram sampling

    :param argsin: tuple of x values, y values, lag cutoff value and interval of x sampling (for easier parallel proc)

    :returns: semivariance
    """

    t_vals, detrend_elev, lag_cutoff, tstep = argsin

    # 1D variogram: inspired by http://connor-johnson.com/2014/03/20/simple-kriging-in-python/

    def SVh(P, h, bw):
        # empirical variogram for a single lag
        pd = np.abs(np.subtract.outer(P[:, 0], P[:, 0]))
        N = pd.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i + 1, N):
                if (pd[i, j] >= h - bw) and (pd[i, j] <= h + bw):
                    Z.append((P[i, 1] - P[j, 1]) ** 2.0)
        if len(Z) > 0:
            return np.array(Z)
        else:
            return np.array([np.nan])

    def SV(P, hs, bw):
        # empirical variogram for a collection of lags
        sv = list()
        for h in hs:
            svh = SVh(P, h, bw)
            sv.append(svh)
        sv = [sv[i] for i in range(len(hs))]
        return sv

    ind_valid = ~np.isnan(detrend_elev)
    sample = np.column_stack((t_vals[ind_valid], detrend_elev[ind_valid]))

    hs = np.arange(0, lag_cutoff, tstep)
    sv = SV(sample, hs, tstep)

    return sv


def wrapper_vgm1d_med(argsin):
    t_vals, stack_detrend_elev, k, lag_cutoff, tstep = argsin

    _, nb_iter = np.shape(stack_detrend_elev)

    print('Pack of ' + str(nb_iter) + ' variograms number ' + str(k))

    lags = np.arange(0, lag_cutoff, tstep)

    vlist = list()
    for i in np.arange(nb_iter):
        sv = vgm_1d_med((t_vals, stack_detrend_elev[:, i], lag_cutoff, tstep))
        vlist.append(sv)

    vzipped = list(zip(*vlist))

    vdata = []
    for i in range(len(vlist[0])):
        vdata.append(np.concatenate(vzipped[i]))

    return vdata


def wrapper_mask(argsin):
    mask, arr_mask, cube_mask = argsin

    out_mask = np.logical_and.reduce(
        (mask, cube_mask, arr_mask[None, :, :] * np.ones(np.shape(mask)[0], dtype=bool)[:, None, None]))

    return out_mask


def wrapper_estimate_var(argsin):
    dh, bins = argsin

    print('Binning slope: ' + str(bins[0][0]) + ' to ' + str(bins[0][1])
          + ' degrees and binning correlation: ' + str(bins[1][0]) + ' to ' + str(bins[1][1]) + ' percent.')

    start = time.time()
    nsamp = 10000
    print('Selecting a subsample of ' + str(nsamp) + ' points...')

    # sample a subset (usually on stable terrain)
    max_samp = len(dh)
    final_nsamp = min(max_samp, nsamp)
    subset = np.random.choice(max_samp, final_nsamp, replace=False)
    sub_dh = dh[subset]

    nmd = nmad(sub_dh)
    ns = final_nsamp
    std = np.nanstd(sub_dh)

    print('Elapsed for bin: ' + str(time.time() - start))

    return nmd, ns, std


def get_var_by_corr_slope_bins(ds, ds_arr, arr_slope, bins_slope, cube_corr, bins_corr, outfile, inc_mask=None,
                               exc_mask=None, nproc=1):
    """
    Sampling method for elevation measurement error from stacks of DEMs: binning of external variables and sampling of variance

    :param ds: xarray Dataset of data cube
    :param ds_arr: Data cube of elevations
    :param arr_slope: Slope raster
    :param bins_slope: Bins to sample categories of slope
    :param cube_corr: Data cube of stereo-correlations
    :param bins_corr: Bins to sample categories of stereo-correlations
    :param outfile: Filename for output files
    :param inc_mask: Filename of inclusion shapefile for the sampling of variance
    :param exc_mask: Filename of exclusion shapefile for the sampling of variance
    :param nproc: Number of cores for multiprocessing [1]

    :returns:
    """

    ref_arr = ds.variables['ref_z'].values

    mask = np.ones(np.shape(ref_arr), dtype=bool)
    if inc_mask is not None:
        mask = np.logical_and(get_stack_mask(inc_mask, ds), mask)
    if exc_mask is not None:
        mask = np.logical_and(~get_stack_mask(exc_mask, ds), mask)

    dh = ds_arr - ref_arr[None, :, :]

    mask_init = np.logical_and(mask[None, :, :] * np.ones(np.shape(dh)[0], dtype=bool)[:, None, None], np.isfinite(dh))

    list_arr_mask = [np.logical_and(np.abs(arr_slope) >= bins_slope[i], np.abs(arr_slope) < bins_slope[i + 1]) for i in
                     range(len(bins_slope) - 1)]
    list_cube_mask = [np.logical_and(np.abs(cube_corr) >= bins_corr[j], np.abs(cube_corr) < bins_corr[j + 1]) for j in
                      range(len(bins_corr) - 1)]

    if nproc == 1:
        print('Using 1 proc to derive variance...')
        list_nmad, list_ns, list_std, list_bin_slope, list_bin_corr, list_id = ([] for i in range(6))
        for i in range(len(bins_slope) - 1):
            for j in range(len(bins_corr) - 1):
                print('Binning slope: ' + str(bins_slope[i]) + ' to ' + str(bins_slope[i + 1])
                      + ' degrees and binning correlation: ' + str(bins_corr[j]) + ' to ' + str(
                    bins_corr[j + 1]) + ' percent.')

                slope_mask = list_arr_mask[i]
                corr_mask = list_cube_mask[j]

                nmd, ns, std = estimate_var(dh, mask_init, arr_mask=slope_mask, cube_mask=corr_mask, nsamp=10000)

                list_nmad.append(nmd)
                list_ns.append(ns)
                list_std.append(std)
                list_id.append('slope: ' + str(bins_slope[i]) + '-'
                               + str(bins_slope[i + 1]) + ',corr:' + str(bins_corr[j]) + '-' + str(bins_corr[j + 1]))
                list_bin_slope.append(bins_slope[i] + (bins_slope[i + 1] - bins_slope[i]) / 2)
                list_bin_corr.append(bins_corr[j] + (bins_corr[j + 1] - bins_corr[j]) / 2)
    else:
        # TODO: does not work, even with np.copy()...

        print('Using ' + str(nproc) + ' procs to derive variance...')
        pool = mp.Pool(nproc, maxtasksperchild=1)

        # print('Creating bin masks...')
        # argsin_mask = [(mask_init,list_arr_mask[i],list_cube_mask[j]) for i in range(len(bins_slope)-1) for j in range(len(bins_corr)-1)]
        # list_mask = pool.map(wrapper_mask,argsin_mask,chunksize=1)
        # pool.close()
        # pool.join()
        # pool = mp.Pool(nproc,maxtasksperchild=1)

        list_mask = [np.logical_and.reduce((mask_init, list_cube_mask[j],
                                            list_arr_mask[i][None, :, :] * np.ones(np.shape(dh)[0], dtype=bool)[:, None,
                                                                           None])) for i in range(len(bins_slope) - 1)
                     for j in range(len(bins_corr) - 1)]

        list_dh = [dh[m] for m in list_mask]
        list_bins = [((bins_slope[i], bins_slope[i + 1]), (bins_corr[j], bins_corr[j + 1])) for i in
                     range(len(bins_slope) - 1) for j in range(len(bins_corr) - 1)]
        argsin = [(list_dh[i], list_bins[i]) for i in range(len(list_dh))]
        print('Deriving variance...')
        outputs = pool.map(wrapper_estimate_var, argsin, chunksize=1)
        pool.close()
        pool.join()

        zipped = list(zip(*outputs))

        list_nmad = zipped[0]
        list_ns = zipped[1]
        list_std = zipped[2]
        list_id = ['slope: ' + str(bins_slope[i]) + '-' + str(bins_slope[i + 1]) + ',corr:' + str(bins_corr[j]) + '-'
                   + str(bins_corr[j + 1]) for i in range(len(bins_slope) - 1) for j in range(len(bins_corr) - 1)]
        list_bin_slope = [(bins_slope[i] + (bins_slope[i + 1] - bins_slope[i]) / 2) for i in range(len(bins_slope) - 1)
                          for j in range(len(bins_corr) - 1)]
        list_bin_corr = [(bins_corr[j] + (bins_corr[j + 1] - bins_corr[j]) / 2) for i in range(len(bins_slope) - 1) for
                         j in range(len(bins_corr) - 1)]

    df = pd.DataFrame()
    df = df.assign(bin_slope=list_bin_slope, bin_corr=list_bin_corr, nmad=list_nmad, std=list_std, nsamp=list_ns,
                   id=list_id)

    df.to_csv(outfile)


def get_var_by_bin(ds, ds_arr, arr_vals, bin_vals, outfile, inc_mask=None, exc_mask=None, rast_cube_mask=False):
    ref_arr = ds.variables['ref_z'].values

    dh = ds_arr - ref_arr[None, :, :]

    mask = np.ones(np.shape(ref_arr), dtype=bool)
    if inc_mask is not None:
        mask = np.logical_and(get_stack_mask(inc_mask, ds), mask)
    if exc_mask is not None:
        mask = np.logical_and(~get_stack_mask(exc_mask, ds), mask)

    mask_init = np.logical_and(mask[None, :, :] * np.ones(np.shape(dh)[0], dtype=bool)[:, None, None], np.isfinite(dh))

    df_all = pd.DataFrame()
    for i in range(len(bin_vals) - 1):
        print('Binning from ' + str(bin_vals[i]) + ' to ' + str(bin_vals[i + 1]))

        if not rast_cube_mask:
            arr_mask = np.logical_and(np.abs(arr_vals) >= bin_vals[i], np.abs(arr_vals) < bin_vals[i + 1])
            cube_mask = None
        else:
            arr_mask = None
            cube_mask = np.logical_and(np.abs(arr_vals) >= bin_vals[i], np.abs(arr_vals) < bin_vals[i + 1])

        nmd, ns, std = estimate_var(dh, mask_init, arr_mask=arr_mask, cube_mask=cube_mask, nsamp=10000)

        bin_id = str(bin_vals[i]) + '-' + str(bin_vals[i + 1])
        df = pd.DataFrame()
        df = df.assign(nmad=[nmd], std=[std], bin_val=[bin_vals[i] + 0.5 * (bin_vals[i + 1] - bin_vals[i])], nsamp=[ns],
                       id=[bin_id])
        df_all = df_all.append(df, ignore_index=True)

    df_all.to_csv(outfile)


def estimate_var(dh, mask_cube, arr_mask=None, cube_mask=None, nsamp=100000):
    """
    Estimation of elevation variance from data cube of difference to reference elevations

    :param dh: Data cube of difference between stack of DEMs and reference DEM
    :param mask_cube: Mask where sampling must occur
    :param arr_mask: Add a raster mask (no time dimension)
    :param cube_mask: Add a data cube mask
    :param nsamp: Number of samples to draw

    :returns: NMAD, number of samples, STD
    """

    start = time.time()

    # rasterize mask
    if arr_mask is not None:
        mask_cube = np.logical_and(mask_cube,
                                   arr_mask[None, :, :] * np.ones(np.shape(dh)[0], dtype=bool)[:, None, None])

    if cube_mask is not None:
        mask_cube = np.logical_and(mask_cube, cube_mask)

    print('Selecting a subsample of ' + str(nsamp) + ' points...')
    # sample a subset (usually on stable terrain)
    max_samp = np.count_nonzero(mask_cube)
    index = np.where(mask_cube)
    final_nsamp = min(max_samp, nsamp)
    subset = np.random.choice(max_samp, final_nsamp, replace=False)
    index_subset = (index[0][subset], index[1][subset], index[2][subset])
    mask_subset = np.zeros(np.shape(mask_cube), dtype=np.bool)
    mask_subset[index_subset] = True

    sub_dh = dh[mask_subset]

    print('Elapsed for bin: ' + str(time.time() - start))

    return nmad(sub_dh), final_nsamp, np.nanstd(sub_dh)


def manual_refine_sampl_temporal_vgm(fn_stack, fn_ref_dem, out_dir, filt_ref='both', max_dhdt=[-50, 50]
                                     , ref_dem_date=np.datetime64('2015-01-01'), inc_mask=None, gla_mask=None, nproc=1):
    """
    Full sampling method of temporal variogram from stacks of DEMs: pre-filtering of data, binning of external variables and sampling of temporal variograms

    :param fn_stack: Filename of netcdf stack of DEMs
    :param fn_ref_dem: Filename of reference DEM
    :param out_dir: Output directory
    :param filt_ref: Filtering method
    :param max_dhdt: Maximum positive/negative elevation change rate per year from reference elevations [-50 m,50 m]
    :param ref_dem_date: Date of reference DEM
    :param inc_mask: Filename of inclusion shapefile for the stack
    :param gla_mask: Filename of inclusion shapefile for the sampling of variogram
    :param nproc: Number of cores for multiprocessing [1]

    :returns:
    """

    # let's look at a few possible dependencies for this temporal vgm
    print('Working on ' + fn_stack)

    ds = xr.open_dataset(fn_stack)
    ds.load()
    start = time.time()

    print('Original temporal size of stack is ' + str(ds.time.size))
    print('Original spatial size of stack is ' + str(ds.x.size) + ',' + str(ds.y.size))

    if inc_mask is not None:
        # ds_orig = ds.copy()
        ds = isel_maskout(ds, inc_mask)
        if ds is None:
            print('Inclusion mask has no valid pixels in this extent. Skipping...')
            return
        print('Temporal size of stack is now: ' + str(ds.time.size))
        print('Spatial size of stack is now: ' + str(ds.x.size) + ',' + str(ds.y.size))

    print('Filtering with max RMSE of 20...')

    keep_vals = ds.uncert.values < 20
    ds = ds.isel(time=keep_vals)
    print('Temporal size of stack is now: ' + str(ds.time.size))

    ds_arr = ds.z.values
    t_vals = ds.time.values

    # pre-filtering
    if fn_ref_dem is not None:
        assert filt_ref in ['min_max', 'time', 'both'], "fn_ref must be one of: min_max, time, both"
        ds_arr_filt = prefilter_stack(ds, ds_arr, fn_ref_dem, t_vals, filt_ref=filt_ref, ref_dem_date=ref_dem_date,
                                      max_dhdt=max_dhdt, nproc=nproc)
        print('Elapsed time is ' + str(time.time() - start))

    fn_final = os.path.join(os.path.dirname(fn_stack), os.path.splitext(os.path.basename(fn_stack))[0] + '_final.nc')
    fn_dh = os.path.join(os.path.dirname(fn_stack), os.path.splitext(os.path.basename(fn_stack))[0] + '_final_dh.tif')
    # TODO: remove when all those are corrected
    if not os.path.exists(fn_dh):
        fn_dh = os.path.join(os.path.dirname(fn_stack),
                             os.path.splitext(os.path.basename(fn_stack))[0] + '_final.nc_dh.tif')

    ds_final = xr.open_dataset(fn_final)
    arr_slope = ds_final.slope.values
    arr_dh = GeoImg(fn_dh).img
    tmp_geo = st.make_geoimg(ds)
    tmp_dem = GeoImg(fn_ref_dem)
    ref_dem = tmp_dem.reproject(tmp_geo).img

    bins_slope = np.arange(0, 60, 10)
    bins_dh = [-300, -200, -100, -50, -20, -10, 0, 10, 20, 50]
    # bins_dh=[-20,-10]
    bins_elev = np.arange(np.nanmin(ref_dem) - np.nanmin(ref_dem) % 200, np.nanmax(ref_dem), 200)

    fn_slope_tvar = os.path.join(out_dir, os.path.splitext(os.path.basename(fn_stack))[0] + '_slope_tvar.csv')
    fn_dh_tvar = os.path.join(out_dir, os.path.splitext(os.path.basename(fn_stack))[0] + '_dh_tvar.csv')
    fn_elev_tvar = os.path.join(out_dir, os.path.splitext(os.path.basename(fn_stack))[0] + '_elev_tvar.csv')

    get_vgm_by_bin(ds, ds_arr, arr_slope, bins_slope, fn_slope_tvar, inc_mask=gla_mask, nproc=nproc)
    get_vgm_by_bin(ds, ds_arr, ref_dem, bins_elev, fn_elev_tvar, inc_mask=gla_mask, nproc=nproc)
    get_vgm_by_bin(ds, ds_arr_filt, arr_dh, bins_dh, fn_dh_tvar, inc_mask=gla_mask, nproc=nproc)


def get_vgm_by_bin(ds, ds_arr, arr_vals, bin_vals, outfile, inc_mask=None, exc_mask=None, nproc=1):
    df_all = pd.DataFrame()
    for i in range(len(bin_vals) - 1):
        print('Binning from ' + str(bin_vals[i]) + ' to ' + str(bin_vals[i + 1]))

        add_mask = np.logical_and(arr_vals >= bin_vals[i], arr_vals < bin_vals[i + 1])
        lags, vmean, vstd = estimate_vgm(ds, ds_arr, inc_mask=inc_mask, exc_mask=exc_mask, rast_mask=add_mask,
                                         nproc=nproc, nsamp=10000)
        id = str(bin_vals[i]) + '-' + str(bin_vals[i + 1])

        df = pd.DataFrame()
        df = df.assign(lags=lags, vmean=vmean, vstd=vstd)
        df = df.assign(id=[id] * len(df.index),
                       bin_val=[bin_vals[i] + 0.5 * (bin_vals[i + 1] - bin_vals[i])] * len(df.index))
        df_all = df_all.append(df, ignore_index=True)

    df_all.to_csv(outfile)


def estimate_vgm(ds, ds_arr, inc_mask=None, exc_mask=None, rast_mask=None, nsamp=10000, tstep=0.25, lag_cutoff=None,
                 min_obs=8, nproc=1, pack_size=50):
    """
    Aggregate 1D temporal variograms for multiple stack pixels: random sampling within a inclusion mask

    :param ds: xarray Dataset of data cube
    :param ds_arr: Data cube of elevations
    :param inc_mask: Filename of inclusion shapefile
    :param exc_mask: Filename of exclusion shapefile
    :param rast_mask: Additional mask (boolean raster)
    :param nsamp: Number of pixels sampled in time
    :param tstep: Time interval for aggregating pairwise time lags
    :param lag_cutoff: Maximum time lag to sample
    :param min_obs: Minimum of valid observation per pixel for sampling
    :param nproc: Number of cores for multiprocessing [1]

    :returns: Lags, Variance, Variance dispersion
    """

    # estimate 1D variogram for multiple pixels: random sampling within mask

    # rasterize mask
    mask = np.ones(np.shape(ds_arr[0]), dtype=bool)

    if inc_mask is not None:
        mask = np.logical_and(get_stack_mask(inc_mask, ds), mask)

    if exc_mask is not None:
        mask = np.logical_and(~get_stack_mask(exc_mask, ds), mask)

    if rast_mask is not None:
        mask = np.logical_and(rast_mask, mask)

    # count number of valid temporal observation for each pixel
    nb_arr = np.nansum(~np.isnan(ds_arr), axis=0)
    mask = np.logical_and(mask, nb_arr >= min_obs)

    print(
        'Selecting a subsample of ' + str(nsamp) + ' points with at least ' + str(min_obs) + ' observations in time...')
    # sample a subset
    max_samp = np.count_nonzero(mask)
    index = np.where(mask)
    final_nsamp = min(max_samp, nsamp)
    subset = np.random.choice(max_samp, final_nsamp, replace=False)
    index_subset = (index[0][subset], index[1][subset])
    mask_subset = np.zeros(np.shape(mask), dtype=np.bool)
    mask_subset[index_subset] = True

    ds_samp = ds_arr[:, mask_subset]
    # ds_tmp = ds.copy()
    # ds_tmp.z.values = ds_arr
    # ds_samp = ds_tmp.isel(y=index[0][subset],x=index[1][subset])['z'].values

    # read and convert time values
    t_vals = ds['time'].values

    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in t_vals])
    t_scale = (ftime_delta / total_delta) * (int(y1) - y0)

    if lag_cutoff is None:
        lag_cutoff = np.max(t_scale) - np.min(t_scale)

    lags = np.arange(0, lag_cutoff, tstep) + 0.5 * tstep

    # get variance/lags and number of pairwise/lags for each pixel

    # old method, changing for median more robust to outliers
    # vdata = np.zeros((len(lags), final_nsamp))
    # pdata = np.zeros((len(lags), final_nsamp))
    if nproc == 1:
        print('Drawing variograms with 1 core...')
        for i in np.arange(final_nsamp):
            # old method, changing for median more robust to outliers
            # sv = vgm_1d((t_scale, ds_samp[:,i].flatten(),lag_cutoff,tstep))
            # vdata[:,i]=sv[0]
            # pdata[:,i]=sv[1]
            vdata = vgm_1d((t_scale, ds_samp[:, i].flatten(), lag_cutoff, tstep))
    else:
        print('Drawing variograms with ' + str(nproc) + ' cores...')
        pool = mp.Pool(nproc, maxtasksperchild=1)
        argsin = [(t_scale, ds_samp[:, i:min(i + pack_size, final_nsamp)], k, lag_cutoff, tstep) for k, i in
                  enumerate(np.arange(0, final_nsamp, pack_size))]
        outputs = pool.map(wrapper_vgm1d_med, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        vdata = []
        if len(outputs) > 0:
            for i in range(len(outputs[0])):
                vdata.append(np.concatenate(zip_out[i]))

        # old method, changing for median more robust to outliers
        # for k, i in enumerate(np.arange(0,final_nsamp,pack_size)):
        #     vdata[:, i:min(i+pack_size,final_nsamp)]=zip_out[0][k]
        #     pdata[:, i:min(i+pack_size,final_nsamp)]=zip_out[1][k]

    # ptot = np.nansum(pdata,axis=1)
    # mean variogram accounting for the number of pairwise comparison in each pixel
    # vmean = np.nansum(vdata * pdata,axis=1) / ptot
    # vmean = np.nanmedian(vdata,axis=1)
    # 'rough' std: between pixels, not accounting for the number of pairwise observation
    # vstd = np.nanstd(vdata,axis=1)

    vmean = np.zeros(len(lags)) * np.nan
    vstd = np.zeros(len(lags)) * np.nan
    if len(vdata) > 0:
        for i in range(len(lags)):
            vmean[i] = np.nanmedian(vdata[i])
            vstd[i] = nmad(vdata[i])

    return lags, vmean, vstd


def plot_vgm(lags, vmean, vstd):
    fig, ax = plt.subplots(1)
    ax.plot(lags, vmean, lw=2, label='mean', color='blue')
    ax.fill_between(lags, vmean + vstd, vmean - vstd, facecolor='blue', alpha=0.5)
    ax.set_title('Variogram: ')
    ax.set_xlabel('Lag [year]')
    ax.set_ylabel('Semivariance [$\mu$ $\pm \sigma$]')
    ax.legend(loc='lower left')
    ax.grid()
    # plt.savefig(fn_fig, dpi=600)
    plt.show()


def parse_epsg(wkt):
    return int(''.join(filter(lambda x: x.isdigit(), wkt.split(',')[-1])))


def ols_matrix(X, Y, conf_interv=0.68, conf_slope=0.68):
    """
    Ordinary Least-Squares (matrix, for optimized processing time)

    :param X: Data cube of x
    :param Y: Data cube of y
    :param conf_interv: Confidence interval for y output
    :param conf_slope: Confidence interval for WLS slope output

    :returns: Slope, Intercept, Slope CI, Y lower CB, Y upper CB
    """
    # perform independent OLS matricially for optimal processing time
    # inspired from: https://en.wikipedia.org/wiki/Simple_linear_regression#Normality_assumption
    # and https://fr.wikipedia.org/wiki/M%C3%A9thode_des_moindres_carr%C3%A9s
    x = X * 1.0
    y = Y * 1.0

    x[np.isnan(y)] = np.nan  # check for NaNs
    y[np.isnan(x)] = np.nan  # check for NaNs

    moy_X = np.nanmean(x, axis=0)
    moy_Y = np.nanmean(y, axis=0)

    mat_cross_product = (x - moy_X) * (y - moy_Y)
    sum_mat_cross_product = np.nansum(mat_cross_product, axis=0)

    mat_X_squared = (x - moy_X) ** 2
    sum_mat_X_squared = np.nansum(mat_X_squared, axis=0)

    beta1 = sum_mat_cross_product / sum_mat_X_squared
    beta0 = moy_Y - beta1 * moy_X

    # confidence interval
    alpha_interv = 1. - conf_interv
    alpha_slope = 1. - conf_slope

    Y_pred = beta1 * x + beta0
    n = np.sum(~np.isnan(x), axis=0)
    SSX = sum_mat_X_squared
    SXY = np.sqrt(np.nansum((y - Y_pred) ** 2, axis=0) / (n - 2))
    SE_slope = SXY / np.sqrt(SSX)
    hi = 1. / n + (x - moy_X) ** 2 / SSX

    # quantile of student's t distribution for p=1-alpha/2
    q_interv = stats.t.ppf(1. - alpha_interv / 2, n - 2)
    q_slope = stats.t.ppf(1. - alpha_slope / 2, n - 2)

    # upper and lower CI:
    dy = q_interv * SXY * np.sqrt(hi)
    Yl = Y_pred - dy
    Yu = Y_pred + dy

    # incert on slope
    incert_slope = q_slope * SE_slope

    return beta1, beta0, incert_slope, Yl, Yu


def wls_matrix(x, y, w, conf_interv=0.68, conf_slope=0.68):
    """
    Weighted Least-Squares (matrix, for optimized processing time)

    :param x: Data cube of x
    :param y: Data cube of y
    :param w: Data cube of weights
    :param conf_interv: Confidence interval for y output
    :param conf_slope: Confidence interval for WLS slope output

    :returns: Slope, Intercept, Slope CI, Y lower CB, Y upper CB
    """

    X = x * 1.0
    Y = y * 1.0
    W = w * 1.0

    Y[np.isnan(W) | np.isnan(X)] = np.nan  # check for NaNs
    X[np.isnan(W) | np.isnan(Y)] = np.nan  # check for NaNs
    W[np.isnan(Y) | np.isnan(X)] = np.nan  # check for NaNs

    sum_w = np.nansum(W, axis=0)
    moy_X_w = np.nansum(X * W, axis=0) / sum_w
    moy_Y_w = np.nansum(Y * W, axis=0) / sum_w

    mat_cross_product = W * (X - moy_X_w) * (Y - moy_Y_w)
    sum_mat_cross_product = np.nansum(mat_cross_product, axis=0)

    mat_X_squared = W * (X - moy_X_w) ** 2
    sum_mat_X_squared = np.nansum(mat_X_squared, axis=0)

    beta1 = sum_mat_cross_product / sum_mat_X_squared
    beta0 = moy_Y_w - beta1 * moy_X_w

    # confidence interval
    alpha_interv = 1. - conf_interv
    alpha_slope = 1. - conf_slope

    Y_pred = beta1 * X + beta0
    n = np.sum(~np.isnan(X), axis=0)
    SSX = sum_mat_X_squared
    SXY = np.sqrt(np.nansum(W * (Y - Y_pred) ** 2, axis=0) / (n - 2))
    SE_slope = SXY / np.sqrt(SSX)
    hi = 1. / n + W * (X - moy_X_w) ** 2 / SSX

    # quantile of student's t distribution for p=1-alpha/2
    q_interv = stats.t.ppf(1. - alpha_interv / 2, n - 2)
    q_slope = stats.t.ppf(1. - alpha_slope / 2, n - 2)

    # get the upper and lower CI:
    dy = q_interv * SXY * np.sqrt(hi)
    Yl = Y_pred - dy
    Yu = Y_pred + dy

    # calculate incert on slope
    incert_slope = q_slope * SE_slope

    return beta1, beta0, incert_slope, Yl, Yu


def interp_data(t, y, sig, interp_t):
    y_ = interp1d(t, y)
    s_ = interp1d(t, sig)
    return y_(interp_t), s_(interp_t)


def robust_wls(t_vals, data, ferr):
    n_out = 1
    while n_out > 0:
        beta1, beta0, incert_slope, _, _ = wls_matrix(t_vals, data, 1. / ferr,
                                                      conf_slope=0.99)
        trend = beta1 * t_vals + beta0
        std_nmad_rat = np.std(data - trend) / nmad(data - trend)
        if std_nmad_rat > 20:
            isin = np.abs(data - trend) < 4 * nmad(data - trend)
        else:
            isin = np.abs(data - trend) < 4 * np.std(data - trend)
        n_out = np.count_nonzero(~isin)

        data = data[isin]
        t_vals = t_vals[isin]
        ferr = ferr[isin]

    return beta1, incert_slope


def detrend(t_vals, data, ferr):
    n_out = 1
    while n_out > 0:
        reg = LinearRegression().fit(t_vals.reshape(-1, 1), data.reshape(-1, 1))

        trend = reg.predict(t_vals.reshape(-1, 1)).squeeze()
        std_nmad_rat = np.std(data - trend) / nmad(data - trend)
        if std_nmad_rat > 20:
            isin = np.abs(data - trend) < 4 * nmad(data - trend)
        else:
            isin = np.abs(data - trend) < 4 * np.std(data - trend)
        n_out = np.count_nonzero(~isin)

        data = data[isin]
        t_vals = t_vals[isin]
        ferr = ferr[isin]

    return reg


def gpr(data, t_vals, uncert, t_pred, opt=False, kernel=None, not_stable=True, detrend_ls=False, loop_detrend=False):
    """
    Gaussian Process regression wrapper

    :param data: Data cube of elevations
    :param t_vals: Time vector of data cube
    :param uncert: Data cube of errors
    :param t_pred: Time vector to predict
    :param opt: Boolean, run kernel optimization
    :param kernel: Provide kernel object
    :param not_stable: Mask of unstable terrain (to apply different kernels)
    :param detrend_ls: Boolean, remove linear trend before applying GP
    :param loop_detrend: Boolean, remove linear trend at each iteration before applying GP

    :returns: Data cube of fitted elevations, Data cube of propagated 1-sigma elevation errors, Data cube mask of valid data
    """

    # if only 0 or 1 elevation values in the pixel, no fitting
    if np.count_nonzero(np.isfinite(data)) < 2:
        return np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(data.shape)

    data_vals = data[np.isfinite(data)]
    err_vals = uncert[np.isfinite(data)] ** 2
    time_vals = t_vals[np.isfinite(data)]

    # by default, no optimizer: applying GPR with defined kernels
    if opt:
        optimizer = 'fmin_l_bfgs_b'
        n_restarts_optimizer = 9
    else:
        optimizer = None
        n_restarts_optimizer = 0

    # initializing
    n_out = 1
    final_fit = 0
    niter = 0
    tag_detr = 1

    num_finite = data_vals.size
    good_vals = np.isfinite(data_vals)
    max_z_score = [20, 12, 9, 6, 4]

    while (n_out > 0 or final_fit == 0) and num_finite >= 2:

        # default kernels
        if kernel is None:

            # weighted least squares
            beta1, beta0, incert_slope, _, _ = wls_matrix(time_vals[good_vals], data_vals[good_vals],
                                                          1. / err_vals[good_vals], conf_slope=0.99)

            # standardized dispersion from linearity (standardized RMSE)
            if ~np.isnan(beta1) and ~np.isnan(beta0):
                res_stdized = np.sqrt(np.mean(
                    (data_vals[good_vals] - (beta0 + beta1 * time_vals[good_vals])) ** 2 / err_vals[good_vals]))
                res = np.sqrt(np.mean((data_vals[good_vals] - (beta0 + beta1 * time_vals[good_vals])) ** 2))
            else:
                res_stdized = 1
                res = np.sqrt(50.)

            # split two periods to try to detect and not filter out surges
            ind_first = np.logical_and(good_vals, time_vals < 10.)
            ind_last = np.logical_and(good_vals, time_vals >= 10.)

            if final_fit == 0:
                nonlin_var = 0
                period_nonlinear = 20
                # first iteration, let it filter out very large outliers
                if niter == 0:
                    base_var = 100.
                    # adapt variance to try not to filter out surges before the final fit with local linear kernel
                elif np.count_nonzero(ind_first) >= 5 and np.count_nonzero(ind_last) >= 5:
                    diff = np.abs(np.mean(data_vals[ind_first]) - np.mean(data_vals[ind_last]))
                    diff_std = np.sqrt(np.std(data_vals[ind_first]) ** 2 + np.std(data_vals[ind_last]) ** 2)
                    if diff - diff_std > 0:
                        base_var = 50 + (diff - diff_std) ** 2 / 2
                    else:
                        base_var = 50.
                else:
                    base_var = 50.
            else:
                # final fit
                base_var = 50.
                if res_stdized != 0:
                    nonlin_var = np.mean(err_vals[good_vals]) + (res / res_stdized) ** 2
                    period_nonlinear = min(100., 100. / res_stdized ** 2)
                else:
                    nonlin_var = np.mean(err_vals[good_vals])
                    period_nonlinear = 100.

            # linear kernel + periodic kernel + local kernel
            k1 = PairwiseKernel(1, metric='linear')  # linear kernel
            k2 = C(30) * ESS(length_scale=1, periodicity=1)  # periodic kernel
            # k3 =  #local kernel
            # k3 = C(50) * RBF(1)
            k3 = C(base_var * 0.6) * RBF(0.75) + C(base_var * 0.3) * RBF(1.5) + C(base_var * 0.1) * RBF(3)
            k4 = PairwiseKernel(1, metric='linear') * C(nonlin_var) * RQ(period_nonlinear, 10)
            kern = k1 + k2
            if not_stable:
                # k3 =  #non-linear kernel
                kern += k3 + k4
        else:
            kern = kernel

        # here we need to change the 0 for the x axis, in case we are using a linear kernel
        mu_x = np.nanmean(time_vals[good_vals])
        detr_t_pred = t_pred - mu_x
        detr_time_vals = time_vals - mu_x
        mu_y = np.nanmean(data_vals)

        if detrend_ls:
            # first, remove a linear trend
            if tag_detr != 0:
                try:
                    # try to remove a linear fit from the data before we fit, then add it back in when we're done.
                    reg = detrend(detr_time_vals[good_vals], data_vals[good_vals], err_vals[good_vals])
                except:
                    return np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(t_pred.shape), np.nan * np.zeros(
                        data.shape)

                l_trend = reg.predict(detr_time_vals.reshape(-1, 1)).squeeze()
                if not loop_detrend:
                    tag_detr = 0

            detr_data_vals = data_vals - l_trend
        else:
            # the mean has to be 0 to do gpr, even if we don't detrend
            detr_data_vals = data_vals - mu_y

        # if we remove a linear trend, normalize_y should be false...
        gp = GaussianProcessRegressor(kernel=kern, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                      alpha=err_vals[good_vals], normalize_y=False)
        gp.fit(detr_time_vals[good_vals].reshape(-1, 1), detr_data_vals[good_vals].reshape(-1, 1))
        y_pred, sigma = gp.predict(detr_t_pred.reshape(-1, 1), return_std=True)
        y_, s_ = interp_data(detr_t_pred, y_pred.squeeze(), sigma.squeeze(), detr_time_vals)
        z_score = np.abs(detr_data_vals - y_) / s_

        isin = z_score[np.isfinite(z_score)] < 4
        # we continue the loop if there is a least one value outside 4 stds
        n_out = np.count_nonzero(~isin)

        # good elevation values can also be outside 4stds because of bias in the first fits
        # thus, if needed, we remove outliers packet by packet, starting with the largest ones
        isout = z_score[np.isfinite(z_score)] > max_z_score[min(niter, len(max_z_score) - 1)]
        tmp_data_vals = data_vals[np.isfinite(z_score)]
        tmp_data_vals[isout] = np.nan
        data_vals[np.isfinite(z_score)] = tmp_data_vals

        good_vals = np.isfinite(data_vals)
        num_finite = np.count_nonzero(good_vals)
        niter += 1

        # no need to filter outliers for final fit
        if final_fit == 1:
            n_out = 0
        # if we have no outliers outside 4 std, initialize back values to jump directly to the final fitting step
        if n_out == 0 and final_fit == 0 and niter > 1:
            n_out = 1
            final_fit = 1
            niter = len(max_z_score) - 1

    # if there is not enough data left...
    if num_finite < 2:
        y_pred = np.nan * np.zeros(t_pred.shape)
        sigma = np.nan * np.zeros(t_pred.shape)

    if detrend_ls:
        l_pred = reg.predict(t_pred.reshape(-1, 1)).squeeze()
        y_out = y_pred.squeeze() + l_pred
    else:
        y_out = y_pred.squeeze() + mu_y

    filt_data = data
    filt_data[np.isfinite(data)] = data_vals

    return y_out, sigma.squeeze(), np.isfinite(filt_data)


def ls(subarr, t_vals, err, weigh, filt_ls=False, conf_filt=0.99):
    """
    Ordinary or Weighted Least-Squares wrapper

    :param subarr: Data cube of elevations
    :param t_vals: Time vector of data cube
    :param err: Data cube of errors
    :param weigh: Boolean, use weighted least-squares
    :param filt_ls: Boolean, use a first CI filtering before fitting again
    :param conf_filt: Confidence interval of the filtering fit

    :returns: Raster concatenation (LS slope, LS intercept, slope 1-sigma error, number of samples, date of first sample, date of last sample), Data cube mask of valid data
    """

    T, Y, X = subarr.shape

    # converting time values
    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in t_vals])
    time_vals = y0 + (ftime_delta / total_delta) * (int(y1) - y0)

    z_mat = subarr.reshape(T, Y * X)
    t_mat = np.array([time_vals, ] * Y * X).T
    w_mat = err.reshape(T, Y * X)
    # old error
    # if weigh:
    #     w_mat = np.array([1. / err ** 2, ] * Y * X).T

    if filt_ls:
        if weigh:
            yl, yu = wls_matrix(t_mat, z_mat, w_mat, conf_interv=conf_filt)[3:5]
        else:
            yl, yu = ols_matrix(t_mat, z_mat, conf_interv=conf_filt)[3:5]

        z_mat[z_mat < yl] = np.nan
        z_mat[z_mat > yu] = np.nan

    if weigh:
        beta1, beta0, incert_slope = wls_matrix(t_mat, z_mat, w_mat)[0:3]
    else:
        beta1, beta0, incert_slope = ols_matrix(t_mat, z_mat)[0:3]

    date_min = np.nanmin(t_mat, axis=0)
    date_max = np.nanmax(t_mat, axis=0)
    nb_trend = (~np.isnan(z_mat)).sum(axis=0)

    filter_less_2_DEMs = nb_trend <= 2
    beta1[filter_less_2_DEMs] = np.nan
    incert_slope[filter_less_2_DEMs] = np.nan

    slope = np.reshape(beta1, (Y, X))
    interc = np.reshape(beta0, (Y, X))
    slope_sig = np.reshape(incert_slope, (Y, X))
    nb_dem = np.reshape(nb_trend, (Y, X))
    date_min = np.reshape(date_min, (Y, X))
    date_max = np.reshape(date_max, (Y, X))

    filt_subarr = np.isfinite(z_mat.reshape(T, Y, X))

    outarr = np.stack((slope, interc, slope_sig, nb_dem, date_min, date_max), axis=0)

    return outarr, filt_subarr


# @jit
def gpr_wrapper(argsin):
    z, i, t_vals, err, new_t, opt, kernel, uns_arr = argsin
    start = time.time()

    Y, X = z.shape[1:3]
    outarr = np.nan * np.zeros((new_t.size * 2 + z.shape[0], Y, X))
    # pixel by pixel
    for x in range(X):
        for y in range(Y):
            if uns_arr is not None:
                uns_tag = uns_arr[0, y, x]
            else:
                uns_tag = True
            uncert = err[:, y, x]
            tmp_y, tmp_sig, tmp_filt = gpr(z[:, y, x], t_vals, uncert, new_t, opt=opt, not_stable=uns_tag,
                                           kernel=kernel)[0:3]
            out = np.concatenate((tmp_y, tmp_sig, tmp_filt), axis=0)
            outarr[:, y, x] = out
    elaps = time.time() - start

    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr


def gpr_dask_wrapper(z, err, t_vals, new_t, opt=False, kernel=None, uns_arr=None):
    start = time.time()

    Y, X = z.shape[0:2]
    outarr = np.nan * np.zeros((Y, X, new_t.size * 2 + z.shape[2],))
    # pixel by pixel
    for x in range(X):
        for y in range(Y):
            if uns_arr is not None:
                uns_tag = uns_arr[y, x, 0]
            else:
                uns_tag = True
            uncert = err[y, x, :]
            tmp_y, tmp_sig, tmp_filt = gpr(z[y, x, :], t_vals, uncert, new_t, opt=opt, not_stable=uns_tag,
                                           kernel=kernel)[0:3]
            out = np.concatenate((tmp_y, tmp_sig, tmp_filt), axis=0)
            outarr[y, x, :] = out
    elaps = time.time() - start

    print('Done with block, elapsed time {}.'.format(elaps))
    return outarr


def gpr_dask(z, err, time_vals, new_t):
    part_fun = functools.partial(gpr_dask_wrapper, t_vals=time_vals, new_t=new_t)

    return xr.apply_ufunc(part_fun, z, err, input_core_dims=[['time'], ['time']], output_core_dims=[['new_time']],
                          output_sizes={'new_time': 2 * len(new_t) + len(time_vals)},
                          dask='parallelized', output_dtypes=[float], join='outer')


@jit
def ls_wrapper(argsin):
    subarr, i, t_vals, err, weigh, filt_ls, conf_filt = argsin
    start = time.time()
    # matrix
    outarr, filt_subarr = ls(subarr, t_vals, err, weigh, filt_ls=filt_ls, conf_filt=conf_filt)
    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr, filt_subarr


def splitter(img, nblocks):
    # manually split a datacube for parallel computing

    split1 = np.array_split(img, nblocks[0], axis=1)
    split2 = [np.array_split(im, nblocks[1], axis=2) for im in split1]
    olist = [np.copy(a) for a in list(chain.from_iterable(split2))]
    return olist


def stitcher(outputs, nblocks):
    # manually stitch a datacube for parallel computing

    stitched = []
    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])
    for i in range(nblocks[0]):
        stitched.append(outputs[i * nblocks[1]])
        for j in range(1, nblocks[1]):
            outind = j + i * nblocks[1]
            stitched[i] = np.concatenate((stitched[i], outputs[outind]), axis=2)
    return np.concatenate(tuple(stitched), axis=1)


def patchify(arr, nblocks, overlap):
    # manually split with overlap a datacube for parallel computing

    overlap = int(np.floor(overlap))
    patches = []
    nx, ny = np.shape(arr)
    nx_sub = nx // nblocks[0]
    ny_sub = ny // nblocks[1]
    split = [[nx_sub * i, min(nx_sub * (i + 1), nx), ny_sub * j, min(ny_sub * (j + 1), ny)] for i in
             range(nblocks[0] + 1) for j in range(nblocks[1] + 1)]
    over = [[max(0, l[0] - overlap), min(nx, l[1] + overlap), max(0, l[2] - overlap), min(l[3] + overlap, ny)] for l in
            split]
    inv = []
    for k in range(len(split)):
        x0, x1, y0, y1 = split[k]
        i0, i1, j0, j1 = over[k]
        patches.append(arr[i0:i1, j0:j1])
        inv.append([x0 - i0, x1 - i0, y0 - j0, y1 - j0])

    return patches, inv, split


def unpatchify(arr_shape, subarr, inv, split):
    # manually stitch account for split overlap a datacube for parallel computing

    out = np.zeros(arr_shape)

    for k, arr in enumerate(subarr):
        s = split[k]
        i = inv[k]
        out[s[0]:s[1], s[2]:s[3]] = arr[i[0]:i[1], i[2]:i[3]]

    return out


def cube_to_stack(ds, out_cube, y0, nice_fit_t, outfile, slope_arr=None, ci=True, clobber=False, filt_bool=False):
    """
    Write data cube to netcdf with new time axis, confidence interval, slope, etc...
    #TODO: this could be simplified quite a bit...

    :param ds: xarray Dataset of data cube
    :param out_cube: Data cube of time series of elevations
    :param y0: Origin of time vector
    :param nice_fit_t: Time vector relative to the origin
    :param outfile: Filename of output netcdf
    :param slope_arr: Slope raster to write as concomitant DataArray
    :param ci: Boolean to look for confidence interval in the out_cube
    :param clobber: Boolean to replace output file
    :param filt_bool: Boolean to force a boolean datacube as output (e.g., to the mask data cube of the multi-step filtering)

    :return:
    """

    fit_cube = out_cube[:len(nice_fit_t), :, :]

    img_shape = np.zeros(np.shape(fit_cube)[1:3])

    nco, to, xo, yo = st.create_nc(img_shape, outfile=outfile,
                                   clobber=clobber, t0=np.datetime64('{}-01-01'.format(y0)))
    st.create_crs_variable(parse_epsg(ds['crs'].spatial_ref), nco)

    x, y = ds['x'].values, ds['y'].values
    xo[:] = x
    yo[:] = y
    to[:] = nice_fit_t

    if not filt_bool:
        dt = 'f4'
        fill = -9999
    else:
        dt = 'i1'
        fill = False
        fit_cube = np.array(fit_cube, dtype=bool)

    zo = nco.createVariable('z', dt, ('time', 'y', 'x'), fill_value=fill, zlib=True,
                            chunksizes=[500, min(150, ds.y.size), min(150, ds.x.size)])
    zo.units = 'meters'
    zo.long_name = 'Fit elevation above WGS84 ellipsoid'
    zo.grid_mapping = 'crs'
    zo.coordinates = 'x y'
    zo.set_auto_mask(True)

    zo[:] = fit_cube

    if ci:
        sig_cube = out_cube[len(nice_fit_t):, :, :]
        fzo = nco.createVariable('z_ci', 'f4', ('time', 'y', 'x'), fill_value=-9999, zlib=True,
                                 chunksizes=[500, min(150, ds.y.size), min(150, ds.x.size)])
        fzo.units = 'meters'
        fzo.long_name = '68% confidence interval for elevation fit.'
        fzo.grid_mapping = 'crs'
        fzo.coordinates = 'x y'
        fzo.set_auto_mask(True)

        fzo[:] = sig_cube

    if slope_arr is not None:
        so = nco.createVariable('slope', 'f4', ('y', 'x'), fill_value=-9999, zlib=True,
                                chunksizes=[min(150, ds.y.size), min(150, ds.x.size)])
        so.units = 'degrees'
        so.long_name = 'median slope used to condition elevation uncertainties'
        so.grid_mapping = 'crs'
        so.coordinates = 'x y'

        so[:] = slope_arr

    nco.close()


def arr_to_img(ds, out_arr, outfile):
    outfile_slope = os.path.join(os.path.dirname(outfile),
                                 os.path.splitext(os.path.basename(outfile))[0] + '_dh.tif')
    outfile_interc = os.path.join(os.path.dirname(outfile),
                                  os.path.splitext(os.path.basename(outfile))[0] + '_interc.tif')
    outfile_sig = os.path.join(os.path.dirname(outfile),
                               os.path.splitext(os.path.basename(outfile))[0] + '_err.tif')
    outfile_nb = os.path.join(os.path.dirname(outfile),
                              os.path.splitext(os.path.basename(outfile))[0] + '_nb.tif')
    outfile_dmin = os.path.join(os.path.dirname(outfile),
                                os.path.splitext(os.path.basename(outfile))[0] + '_dmin.tif')
    outfile_dmax = os.path.join(os.path.dirname(outfile),
                                os.path.splitext(os.path.basename(outfile))[0] + '_dmax.tif')

    arr = st.make_geoimg(ds, band=0)
    arr.img = out_arr[0, :, :]
    arr.write(outfile_slope)
    arr.img = out_arr[1, :, :]
    arr.write(outfile_interc)
    arr.img = out_arr[2, :, :]
    arr.write(outfile_sig)
    arr.img = out_arr[3, :, :]
    arr.write(outfile_nb)
    arr.img = out_arr[4, :, :]
    arr.write(outfile_dmin)
    arr.img = out_arr[5, :, :]
    arr.write(outfile_dmax)


# TODO: homogenize how REF DEM is provided: sometimes filename, sometimes raster, sometimes GeoImg... inadmissible :)

def time_filter_ref(ds_arr, ref_arr, t_vals, ref_date, max_dhdt=[-50, 50], base_thresh=100.):
    """
    Temporal filtering of stacked DEMs with a reference DEM: removes all elevations outside a positive/negative linear
     elevation trend from the reference, with a base threshold.

    :param ds_arr: Data cube of elevations
    :param ref_arr: Raster of reference DEM
    :param t_vals: Time vector of  data cube
    :param ref_date: Date of reference DEM
    :param max_dhdt: Maximum positive/negative elevation change rate per year from reference elevations [-50 m,50 m]
    :param base_thresh: Base vertical threshold for the temporal filter (avoid elevations close to reference) [100 m]

    :return: ds_arr: Filtered data cube
    """

    print('Adding base threshold of ' + str(base_thresh) + ' m around reference values.')

    delta_t = (ref_date - t_vals).astype('timedelta64[D]').astype(float) / 365.24
    dh = ref_arr[None, :, :] - ds_arr
    dt_arr = np.ones(dh.shape) * delta_t[:, None, None]
    # dh = ref_arr[:,:,None] - z_arr
    # dt_arr = np.ones(dh.shape) * delta_t[None,None,:]
    if np.array(max_dhdt).size == 1:
        ds_arr[np.abs(dh) > base_thresh + np.abs(dt_arr) * max_dhdt] = np.nan
    else:
        ds_arr[np.logical_or(np.logical_and(dt_arr < 0, np.logical_or(dh < - base_thresh + dt_arr * max_dhdt[1],
                                                                      dh > base_thresh + dt_arr * max_dhdt[0])),
                             np.logical_and(dt_arr > 0, np.logical_or(dh > base_thresh + dt_arr * max_dhdt[1],
                                                                      dh < - base_thresh + dt_arr * max_dhdt[
                                                                          0])))] = np.nan
    return ds_arr


def dask_time_filter_ref(ds, ds_arr, ref_dem, t_vals, ref_date, max_dhdt=[-50, 50], base_thresh=100., nproc=1):
    """
       Dask wrapper of temporal filtering of stacked DEMs with a reference DEM: removes all elevations outside a
       positive/negative linear elevation trend from the reference, with a base threshold.

       :param ds_arr: Data cube of elevations
       :param ds_arr: Data cube of elevations
       :param ref_dem: GeoImg of reference DEM
       :param t_vals: Time vector of  data cube
       :param ref_date: Date of reference DEM
       :param max_dhdt: Maximum positive/negative elevation change rate per year from reference elevations [-50 m,50 m]
       :param base_thresh: Base vertical threshold for the temporal filter (avoid elevations close to reference) [100 m]
       :param nproc: Number of cores for multiprocessing [1]

       :return: Filtered data cube
       """

    start = time.time()
    print('Setting up time filtering parallelized...')
    part_tf = functools.partial(time_filter_ref, t_vals=t_vals, ref_date=ref_date, max_dhdt=max_dhdt,
                                base_thresh=base_thresh)

    z_dask = xr.DataArray(ds_arr, coords=[ds.time.values, ds.y.values, ds.x.values], dims=['time', 'y', 'x'])
    ref_arr = ref_dem.img
    ref_dask = xr.DataArray(ref_arr, coords=[ds.y.values, ds.x.values], dims=['y', 'x'])

    print('Time filter elapsed time is ' + str(time.time() - start))
    print('Chunking...')
    z_chunk = z_dask.chunk({'x': 200, 'y': 200})
    ref_chunk = ref_dask.chunk({'x': 200, 'y': 200})

    print('Time filter elapsed time is ' + str(time.time() - start))
    print('Chunking done, computing.')

    sl = xr.apply_ufunc(part_tf, z_chunk, ref_chunk, input_core_dims=[['time'], []], output_core_dims=[['time']],
                        dask='parallelized', output_dtypes=[float])
    filt_z_dask = sl.compute(num_workers=nproc, scheduler='processes')
    filt_z_dask = filt_z_dask.transpose('time', 'y', 'x')

    print('Time filter elapsed time is ' + str(time.time() - start))

    return filt_z_dask.values


@jit_filter_function
def nanmax(a):
    return np.nanmax(a)


@jit_filter_function
def nanmin(a):
    return np.nanmin(a)


def wrapper_slope(argsin):
    arr, in_met = argsin

    start = time.time()
    slope_list = []

    # create input image
    gt, proj, npix_x, npix_y = in_met
    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)
    sp = dst.SetProjection(proj)
    sg = dst.SetGeoTransform(gt)
    for i in range(np.shape(arr)[0]):
        out_arr = np.copy(arr[i, :])
        out_arr[np.isnan(out_arr)] = -9999
        wa = dst.GetRasterBand(1).WriteArray(out_arr)
        md = dst.SetMetadata({'Area_or_point': 'Point'})
        nd = dst.GetRasterBand(1).SetNoDataValue(-9999)
        tmp_z = GeoImg(dst)

        slope = get_slope(tmp_z)

        slope_list.append(slope.img)
    del sp, sg, wa, md, nd

    slope_stack = np.stack(slope_list, axis=0)

    print('Deriving slope stack in ' + str(time.time() - start))

    return slope_stack


def maxmin_disk_filter(argsin):
    arr, rad = argsin

    max_arr = filters.generic_filter(arr, nanmax, footprint=disk(rad))
    min_arr = filters.generic_filter(arr, nanmin, footprint=disk(rad))

    return max_arr, min_arr


@jit
def robust_nanmax(a):
    return np.nanpercentile(a, 80)
    # return np.nanmax(a[np.abs(np.nanmedian(a)-a)<3*nmad(a)])


@jit
def robust_nanmin(a):
    # return np.nanmin(a[np.abs(np.nanmedian(a)-a)<3*nmad(a)])
    return np.nanpercentile(a, 20)


def robust_maxmin_disk_filter(argsin):
    arr, rad = argsin

    max_arr = filters.generic_filter(arr, robust_nanmax, footprint=disk(rad))
    min_arr = filters.generic_filter(arr, robust_nanmin, footprint=disk(rad))

    return max_arr, min_arr


def spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=500, cutoff_thr=20., nproc=1):
    """
    Spatial filtering of stacked DEMs with a reference DEM: removes all elevations smaller than the minimum or larger
    than the maximum reference elevation found within a circle of certain size, with a base threshold.

    :param ds_arr: Data cube of elevations
    :param ref_dem: GeoImg of reference DEM
    :param cutoff_kern_size: Radius size for kernel spatial filtering [500 m]
    :param cutoff_thr: Base vertical threshold for the spatial filter (avoid elevations close to reference)

    :return: ds_arr: Filtered data cube
    """

    # here we assume that the reference DEM is a "clean" post-processed DEM, filtered with QA for low confidence outliers
    # minimum/maximum elevation in circular surroundings based on reference DEM

    ref_arr = ref_dem.img
    res = 100.
    rad = int(np.floor(cutoff_kern_size / res))

    if nproc == 1:
        print('Filtering min/max in radius of ' + str(cutoff_kern_size) + 'm, base threshold of ' + str(
            cutoff_thr) + 'm on 1 proc...')
        max_arr, min_arr = maxmin_disk_filter((ref_arr, rad))
    else:
        print('Filtering min/max in radius of ' + str(cutoff_kern_size) + 'm, base threshold of ' + str(
            cutoff_thr) + 'm on ' + str(nproc) + ' procs...')
        nopt = int(np.floor(np.sqrt(nproc)))
        nblocks = [nopt, nopt]
        patches, inv, split = patchify(ref_arr, nblocks, rad)

        pool = mp.Pool(nproc, maxtasksperchild=1)
        argsin = [(p, rad) for i, p in enumerate(patches)]
        outputs = pool.map(maxmin_disk_filter, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        max_arr = unpatchify(np.shape(ref_arr), zip_out[0], inv, split)
        min_arr = unpatchify(np.shape(ref_arr), zip_out[1], inv, split)

    for i in range(ds_arr.shape[0]):
        ds_arr[i, np.logical_or(ds_arr[i, :] > (max_arr + cutoff_thr), ds_arr[i, :] < (min_arr - cutoff_thr))] = np.nan

    return ds_arr


def nanmedian_slope(slope_cube):
    return np.nanmedian(slope_cube, axis=2)


def isel_merge_dupl_dates(ds):
    """
    Merge duplicate dates of a xarray dataset by taking the mean of similar dates (e.g., overlapping same-date ASTER DEMs)

    :param ds: xarray Dataset of datacube
    :return: ds: merged Dataset
    """

    # merge DEMs elevations (np.nanmean) for similar dates
    t_vals = ds.time.values
    dates_rm_dupli = sorted(list(set(list(t_vals))))
    ind_firstdate = []
    for i, date in enumerate(dates_rm_dupli):
        ind_firstdate.append(list(t_vals).index(date))
    ds_filt = ds.isel(time=np.array(ind_firstdate))
    for i in range(len(dates_rm_dupli)):
        t_ind = (t_vals == dates_rm_dupli[i])
        if len(t_ind) > 1:
            ds_filt.z.values[i, :] = np.nanmean(ds.z.values[t_ind, :], axis=0)
            ds_filt.uncert.values[i, :] = np.nanmean(ds.uncert.values[t_ind, :])

            # something is wrong when doing weighted mean...

            # careful, np.nansum gives back zero for an axis full of NaNs
            # mask_nan = np.all(np.isnan(ds.z.values[t_ind,:]),axis=0)
            # ds_filt.z.values[i, :] = np.nansum(ds.z.values[t_ind, :] * 1./ds.uncert.values[t_ind,None,None]**2,
            # axis=0)/ np.nansum(1./ds.uncert.values[t_ind,None,None]**2, axis=0)
            # ds_filt.z.values[i, mask_nan] = np.nan
            # ds_filt.uncert.values[i] = np.nansum(ds.uncert.values[t_ind] * 1./ds.uncert.values[t_ind]**2) /
            # np.nansum(1./ds.uncert.values[t_ind]**2)

    ds = ds_filt

    return ds


def isel_maskout(ds, inc_mask):
    """
    Mask out values out of shapefile and minimize spatial extent, remove void time steps (to decrease computing time)

    :param ds: xarray Dataset of datacube
    :return: ds: reduced Dataset
    """

    # simplify extent to mask, mask out remaining masked pixels in the extent as NaNs, remove unusued time indexes
    land_mask = get_stack_mask(inc_mask, ds)
    if np.count_nonzero(land_mask) > 0:
        ds, submask, slices = tt.sel_dc(ds, None, land_mask)
        print('Including only ' + str(np.count_nonzero(land_mask)) + ' pixels out of ' + str(
            np.shape(land_mask)[0] * np.shape(land_mask)[1]) + ' on this tile')
        ds_arr = ds.z.values
        ds_arr[:, ~submask] = np.nan
        print('Consequently removing void DEMs...')
        non_void = np.count_nonzero(np.isfinite(ds_arr), axis=(1, 2)) > 0
        ds = ds.isel(time=non_void)
    else:
        ds = None

    return ds


def constrain_var_slope_corr(ds, ds_arr, ds_corr, t_vals, uncert, fn_ref_dem=None, nproc=1):
    """
    Given a data cube of stacked DEMs, a data cube of stereo-correlations and a per-DEM vector of co-registration
    RMSE: estimate a median slope and compute a data cube of elevation errors.
    (error dependance on the slope and stereo-correlation is defined independently)
    #TODO: allow for a user-defined function of error?

    :param ds: xarray Dataset of datacube
    :param ds_arr: Data cube of elevations
    :param ds_corr: Data cube of stereo-correlations
    :param t_vals: Time vector of data cube
    :param uncert: Vector of co-registration RMSEs of the stacked DEMs
    :param fn_ref_dem: Filename for input reference DEM
    :param nproc: Number of cores for multiprocessing [1]

    :return: Data cube of errors, Raster of median slope values
    """

    print('>>Starting variance assessment...')

    start_var = time.time()

    if nproc == 1:
        print('Estimating terrain slope to constrain uncertainties with 1 proc...')
        slope_arr = np.zeros(np.shape(ds_arr))
        for i in range(len(t_vals)):
            slope = get_slope(st.make_geoimg(ds, i))
            slope.img[slope.img > 70] = np.nan
            slope_arr[i, :, :] = slope.img
    else:
        print('Estimating terrain slope to constrain uncertainties with ' + str(nproc) + ' procs...')
        tmp_img = st.make_geoimg(ds)
        pack_size = int(np.ceil(ds.time.size / nproc))
        in_met = (tmp_img.gt, tmp_img.proj_wkt, tmp_img.npix_x, tmp_img.npix_y)
        argsin_z = [(ds_arr[i:min(i + pack_size, ds.time.size), :], in_met) for k, i in
                    enumerate(np.arange(0, ds.time.size, pack_size))]
        pool = mp.Pool(nproc, maxtasksperchild=1)
        outputs_z = pool.map(wrapper_slope, argsin_z)
        pool.close()
        pool.join()

        print('Elapsed time during variance assess. is ' + str(time.time() - start_var))
        slope_arr = np.zeros(np.shape(ds_arr))
        for k, i in enumerate(np.arange(0, ds.time.size, pack_size)):
            slope_arr[i:min(i + pack_size, ds.time.size), :] = outputs_z[k]

    # err_dask = xr.DataArray(err_arr, coords=[ds.time.values, ds.y.values, ds.x.values], dims=['time', 'y', 'x'])
    #
    # err_dask = err_dask.chunk({'x': 100, 'y': 100})
    # sl = xr.apply_ufunc(nanmedian_slope, err_dask, input_core_dims=[['time']], dask='parallelized',
    #                     output_dtypes=[float])
    # med_dask = sl.compute(num_workers=nproc, scheduler='processes')
    # med_slope = med_dask.values
    med_slope = np.nanmedian(slope_arr, axis=0)

    if fn_ref_dem is not None:
        tmp_dem = GeoImg(fn_ref_dem)
        slope_all = get_slope(tmp_dem)
        slope_ref = slope_all.reproject(st.make_geoimg(ds))
        med_slope[np.isnan(med_slope)] = slope_ref.img[np.isnan(med_slope)]

    err_arr = np.ones(np.shape(ds_arr), dtype=np.float32)
    err_arr = err_arr * uncert[:, None, None] ** 2

    # based on prior analysis of variance, this is a pretty good generic fit
    slope_err = ((20 + 20 * (100 - ds_corr)) * np.tan(med_slope * np.pi / 180)) ** 2
    med_slope_arr = med_slope[None, :, :] * np.ones(np.shape(ds_arr)[0])[:, None, None]
    slope_err[med_slope_arr > 50] += ((med_slope_arr[med_slope_arr > 50] - 50) * 5) ** 2
    corr_err = (((100 - ds_corr) / 100) * 20) ** 2.5
    err_arr += slope_err
    err_arr += corr_err
    err_arr = np.sqrt(err_arr)
    err_arr[np.logical_or(~np.isfinite(err_arr), np.abs(err_arr) > 300)] = 300

    print('Elapsed time during variance assess. is ' + str(time.time() - start_var))

    return err_arr, med_slope


def prefilter_stack(ds, ds_arr, fn_ref_dem, t_vals, filt_ref='min_max', ref_dem_date=None, max_dhdt=[-50, 50], nproc=1):
    """
    Given a data cube of stacked DEMs and a reference DEM: condition a first-order spatial and temporal
    filtering of outliers.

    :param ds: xarray Dataset of datacube
    :param ds_arr: Data cube of elevations
    :param t_vals: Time vector of data cube
    :param filt_ref: Method of filtering: spatial "min_max", temporal "time" or "both" ["min_max"]
    :param fn_ref_dem: Filename for input reference DEM
    :param ref_dem_date: Date of reference DEM
    :param max_dhdt: Maximum positive/negative elevation change rate per year from reference elevations [-50 m,50 m]
    :param nproc: Number of cores for multiprocessing [1]

    :return: Filtered data cube
    """

    print('>>Starting prefiltering...')
    start_prefilt = time.time()

    print('Number of valid pixels:' + str(np.count_nonzero(~np.isnan(ds_arr))))

    # minimum/maximum elevation on Earth
    ds_arr[np.logical_or(ds_arr < -400, ds_arr > 8900)] = np.nan

    tmp_geo = st.make_geoimg(ds)
    tmp_dem = GeoImg(fn_ref_dem)
    ref_dem = tmp_dem.reproject(tmp_geo)
    if filt_ref == 'min_max':
        print('Filtering spatially using min/max values in {}'.format(fn_ref_dem))
        ds_arr = spat_filter_ref(ds_arr, ref_dem, nproc=nproc)
    elif filt_ref == 'time':
        if ref_dem_date is None:
            print('Reference DEM time stamp not specified, defaulting to 01.01.2000')
            ref_dem_date = np.datetime64('2000-01-01')
        print('Filtering temporally with threshold of {} m/a'.format(max_dhdt))
        # ds_arr =  dask_time_filter_ref(ds, ds_arr, ref_dem, t_vals, ref_dem_date, dhdt_thresh=time_filt_thresh, nproc=nproc)
        ds_arr = time_filter_ref(ds_arr, ref_dem, t_vals, ref_dem_date, max_dhdt=max_dhdt)
    elif filt_ref == 'both':
        print('Filtering spatially using min/max values in {}'.format(fn_ref_dem))
        ds_arr = spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=200, cutoff_thr=700., nproc=nproc)
        ds_arr = spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=500, cutoff_thr=500., nproc=nproc)
        ds_arr = spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=2000, cutoff_thr=300.,
                                 nproc=int(np.floor(nproc / 4)))
        print('Number of valid pixels after spatial filtering:' + str(np.count_nonzero(~np.isnan(ds_arr))))

        print('Elapsed time during prefiltering is ' + str(time.time() - start_prefilt))
        if ref_dem_date is None:
            print('Reference DEM time stamp not specified, defaulting to 01.01.2000')
            ref_dem_date = np.datetime64('2000-01-01')
        print('Filtering temporally with threshold of {} m/a'.format(max_dhdt))
        # ds_arr =  dask_time_filter_ref(ds, ds_arr, ref_dem, t_vals, ref_dem_date, dhdt_thresh=time_filt_thresh, nproc=nproc)
        ds_arr = time_filter_ref(ds_arr, ref_dem.img, t_vals, ref_dem_date, max_dhdt=max_dhdt)
        print('Number of valid pixels after temporal filtering:' + str(np.count_nonzero(~np.isnan(ds_arr))))

        print('Elapsed time during prefiltering is ' + str(time.time() - start_prefilt))

    return ds_arr


def robust_wls_ref_filter_stack(ds, ds_arr, err_arr, t_vals, fn_ref_dem, ref_dem_date=np.datetime64('2013-01-01'),
                                max_dhdt=[-50, 50], nproc=1, cutoff_kern_size=1000, max_deltat_ref=2.,
                                base_thresh=100.):
    """
    Given a data cube of stacked DEMs, of elevation errors and a reference DEM: condition a spatial and temporal
    filtering of outliers based on WLS linear trends.

    :param ds: xarray Dataset of data cube
    :param ds_arr: Data cube of elevations
    :param err_arr: Data cube of elevation errors
    :param t_vals: Time vector of data cube
    :param fn_ref_dem: Filename for input reference DEM
    :param ref_dem_date: Date of reference DEM
    :param max_dhdt: Maximum positive/negative elevation change rate per year from reference elevations [-50 m,50 m]
    :param nproc: Number of cores for multiprocessing [1]
    :param cutoff_kern_size: Radius size for kernel spatial filtering [1000 m]
    :param max_deltat_ref: Maximum time lag between acquisition of reference DEM and date provided [2 yr]
    :param base_thresh: Base vertical threshold for the temporal filter (avoid elevations close to reference) [100 m]

    :return: Filtered data cube
    """

    # TODO: remove ds from the list of inputs, and provide ref_dem_arr instead of fn_ref_dem?

    print('Performing WLS to condition filtering...')

    # wls parameters
    weig = True
    filt_ls = True
    conf_filt_ls = 0.99

    # getting radius size
    tmp_geo = st.make_geoimg(ds)
    tmp_dem = GeoImg(fn_ref_dem)
    ref_dem = tmp_dem.reproject(tmp_geo)
    ref_arr = ref_dem.img
    # TODO: do not hardcore resolution
    res = 100.
    rad = int(np.floor(cutoff_kern_size / res))

    # wls
    if nproc == 1:
        print('Processing with 1 core...')
        out_arr, _ = ls_wrapper((ds_arr, 0, t_vals, err_arr, weig, filt_ls, conf_filt_ls))
    else:
        print('Processing with ' + str(nproc) + ' cores...')
        # here calculation is done matricially so we want to use all cores with the largest tiles possible
        opt_n_tiles = int(np.floor(np.sqrt(nproc)))
        n_x_tiles = opt_n_tiles
        n_y_tiles = opt_n_tiles

        pool = mp.Pool(nproc, maxtasksperchild=1)
        split_arr = splitter(ds_arr, (n_y_tiles, n_x_tiles))
        split_err = splitter(err_arr, (n_y_tiles, n_x_tiles))

        argsin = [(s, i, np.copy(t_vals), np.copy(split_err[i]), weig, filt_ls, conf_filt_ls) for i, s in
                  enumerate(split_arr)]
        outputs = pool.map(ls_wrapper, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))
        out_arr = stitcher(zip_out[0], (n_y_tiles, n_x_tiles))[0, :, :]

    # removing large dhdt outliers
    # print('Writing to rasters for checking...')
    # dh_wls = ref_dem.copy()
    # dh_wls.img = out_arr
    # dh_wls.write('/calcul/malo/hugonnet/dh_wls.tif')
    out_arr[out_arr < max_dhdt[0]] = np.nan
    out_arr[out_arr > max_dhdt[1]] = np.nan

    # finding max/min of dh/dt in a kernel
    if nproc == 1:
        print('Finding min/max dhdt in radius of ' + str(cutoff_kern_size) + 'm on 1 proc...')
        max_dhdt_arr, min_dhdt_arr = robust_maxmin_disk_filter((out_arr, rad))
    else:
        print('Finding min/max dhdt in radius of ' + str(cutoff_kern_size) + 'm on ' + str(nproc) + ' procs...')
        nopt = int(np.floor(np.sqrt(nproc)))
        nblocks = [nopt, nopt]
        patches, inv, split = patchify(out_arr, nblocks, rad)

        pool = mp.Pool(nproc, maxtasksperchild=1)
        argsin = [(p, rad) for i, p in enumerate(patches)]
        outputs = pool.map(robust_maxmin_disk_filter, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        max_dhdt_arr = unpatchify(np.shape(out_arr), zip_out[0], inv, split)
        min_dhdt_arr = unpatchify(np.shape(out_arr), zip_out[1], inv, split)

    # temp: to check visually
    # print('Writing to rasters for checking...')
    # max_dh_img= ref_dem.copy()
    # max_dh_img.img = max_dhdt_arr
    # max_dh_img.write('/calcul/malo/hugonnet/test_max.tif')
    # max_dh_img.img = min_dhdt_arr
    # max_dh_img.write('/calcul/malo/hugonnet/test_min.tif')

    # using max/min dhdt to better condition spatio-temporal filtering from reference
    if nproc == 1:
        print('Finding min/max ref in radius of ' + str(cutoff_kern_size) + 'm on 1 proc...')
        max_ref_arr, min_ref_arr = maxmin_disk_filter((ref_arr, rad))
    else:
        print('Finding min/max ref in radius of ' + str(cutoff_kern_size) + 'm on ' + str(nproc) + ' procs...')
        nopt = int(np.floor(np.sqrt(nproc)))
        nblocks = [nopt, nopt]
        patches, inv, split = patchify(ref_arr, nblocks, rad)

        pool = mp.Pool(nproc, maxtasksperchild=1)
        argsin = [(p, rad) for i, p in enumerate(patches)]
        outputs = pool.map(maxmin_disk_filter, argsin, chunksize=1)
        pool.close()
        pool.join()

        zip_out = list(zip(*outputs))

        max_ref_arr = unpatchify(np.shape(ref_arr), zip_out[0], inv, split)
        min_ref_arr = unpatchify(np.shape(ref_arr), zip_out[1], inv, split)

    max_abs_dhdt = np.nanmax(np.stack((np.abs(min_dhdt_arr), np.abs(max_dhdt_arr))), axis=0)

    # max_dh_img.img = max_abs_dhdt
    # max_dh_img.write('/calcul/malo/hugonnet/max_abs.tif')

    min_dhdt_filt = np.nanmin(np.stack((np.zeros(np.shape(min_dhdt_arr)), min_dhdt_arr)), axis=0)
    max_dhdt_filt = np.nanmax(np.stack((np.zeros(np.shape(max_dhdt_arr)), max_dhdt_arr)), axis=0)

    # max_dh_img.img = min_dhdt_filt
    # max_dh_img.write('/calcul/malo/hugonnet/min_filt.tif')

    # max_dh_img.img = max_dhdt_filt
    # max_dh_img.write('/calcul/malo/hugonnet/max_filt.tif')

    print('Refining spatio-temporal filtering with trend values...')
    # spatial filtering refined with temporal approx of dhdt
    print('Initial valid pixels: ' + str(np.count_nonzero(~np.isnan(ds_arr))))

    for i in range(ds_arr.shape[0]):
        ds_arr[i, np.logical_or(ds_arr[i, :] > (max_ref_arr + base_thresh + 30 * max_abs_dhdt),
                                ds_arr[i, :] < (min_ref_arr - base_thresh - 30 * max_abs_dhdt))] = np.nan

    print('Pixels after refined spatial filtering: ' + str(np.count_nonzero(~np.isnan(ds_arr))))
    # temporal filtering refined with temporal approx of dhdt
    delta_t = (ref_dem_date - t_vals).astype('timedelta64[D]').astype(float) / 365.24
    dh = ref_arr[None, :, :] - ds_arr
    dt_arr = np.ones(dh.shape) * delta_t[:, None, None]
    ds_arr[np.logical_or(np.logical_and(dt_arr < 0, np.logical_or(
        dh < - (base_thresh + max_deltat_ref * max_abs_dhdt[None, :, :]) + dt_arr * 2 * max_abs_dhdt[None, :, :],
        dh > (base_thresh + max_deltat_ref * max_abs_dhdt[None, :, :]) + dt_arr * 2 * (-max_abs_dhdt[None, :, :]))),
                         np.logical_and(dt_arr > 0, np.logical_or(
                             dh > (base_thresh + max_deltat_ref * max_abs_dhdt[None, :, :]) + dt_arr * 2 * max_abs_dhdt[
                                                                                                           None, :, :],
                             dh < - (base_thresh + max_deltat_ref * max_abs_dhdt[None, :, :]) + dt_arr * 2 * (
                                 -max_abs_dhdt[None, :, :]))))] = np.nan

    print('Pixels after refined temporal filtering: ' + str(np.count_nonzero(~np.isnan(ds_arr))))

    return ds_arr


def fit_stack(fn_stack, fit_extent=None, fn_ref_dem=None, ref_dem_date=None, filt_ref='min_max',
              time_filt_thresh=[-30, 30], inc_mask=None, exc_mask=None, nproc=1, method='gpr',
              opt_gpr=False, kernel=None, filt_ls=False, conf_filt_ls=0.99, tlim=None, tstep=0.25, outfile='fit.nc',
              write_filt=False, clobber=False, merge_date=False, dask_parallel=False):
    """
    Given a netcdf stack of DEMs, perform filtering and temporal fitting of elevation with uncertainty propagation

    :param fn_stack: Filename for input netcdf file
    :param fit_extent: extent over which to limit fit, as [xmin, xmax, ymin, ymax]
    :param fn_ref_dem: Filename for input reference DEM
    :param ref_dem_date: Date of reference DEM
    :param filt_ref: Type of filtering. One of 'min_max', 'time', or 'both'. Requires a reference DEM.
    :param time_filt_thresh: Maximum dh/dt from reference DEM for time filtering
    :param inc_mask: Optional inclusion mask. Pixels outside of the mask will not be fit.
    :param exc_mask: Optional exclusion mask. Pixels inside of the mask will not be fit.
    :param nproc: Number of cores for multiprocessing [1]
    :param method: Fitting method, currently supported: Gaussian Process Regression "gpr", Ordinary Least Squares "ols"
     and Weighted Least Squares "wls" ["gpr"]
    :param opt_gpr: Run learning optimization in the GPR fitting [False]
    :param kernel: Kernel
    :param filt_ls: Filter least square with a first fit [False]
    :param conf_filt_ls: Confidence interval to filter least square fit [99%]
    :param tlim: time range [t_min t_max] over which to fit. Default is full range of stack.
    :param tstep: Temporal step for fitted stack [0.25 year]
    :param outfile: Path to outfile
    :param write_filt: Write filtered stack to file
    :param clobber: Overwrite existing output files
    :param merge_date: Merge any DEMs which have the same acquisition date
    :param dask_parallel: run with dask parallel tools
    :return:
    """

    assert method in ['gpr', 'ols', 'wls'], "Method must be one of gpr, ols or wls."
    print('Reading dataset: ' + fn_stack)

    start = time.time()

    ds = xr.open_dataset(fn_stack)
    ds.load()

    if fit_extent is not None:
        xmin, xmax, ymin, ymax = fit_extent
        ds = ds.sel(x=slice(xmin, xmax), y=slice(ymin, ymax))

    print('Original temporal size of stack is ' + str(ds.time.size))
    print('Original spatial size of stack is ' + str(ds.x.size) + ',' + str(ds.y.size))

    if inc_mask is not None:
        # ds_orig = ds.copy()
        ds = isel_maskout(ds, inc_mask)
        if ds is None:
            print('Inclusion mask has no valid pixels in this extent. Skipping...')
            return
        print('Temporal size of stack is now: ' + str(ds.time.size))
        print('Spatial size of stack is now: ' + str(ds.x.size) + ',' + str(ds.y.size))
        print('Elapsed time is ' + str(time.time() - start))

    print('Filtering with max RMSE of 20...')

    keep_vals = ds.uncert.values < 20
    ds = ds.isel(time=keep_vals)
    print('Temporal size of stack is now: ' + str(ds.time.size))

    if merge_date:
        print('Merging ASTER DEMs with exact same date...')
        ds = isel_merge_dupl_dates(ds)

    print('Final temporal size of stack is ' + str(ds.time.size))
    print('Elapsed time is ' + str(time.time() - start))

    if exc_mask is not None:
        uns_mask = get_stack_mask(exc_mask, ds)
        uns_arr = uns_mask[np.newaxis, :, :]
    else:
        uns_arr = None

    ds_arr = ds.z.values
    ds_corr = ds.corr.values
    # change correlation for SETSM segments
    ind_setsm = np.array(['SETSM' in name for name in ds.dem_names.values])
    ds_corr[ind_setsm, :] = 60.
    t_vals = ds.time.values
    uncert = ds.uncert.values
    filt_vals = (t_vals - np.datetime64('2000-01-01')).astype('timedelta64[D]').astype(int)

    # pre-filtering
    if fn_ref_dem is not None:
        assert filt_ref in ['min_max', 'time', 'both'], "fn_ref must be one of: min_max, time, both"
        ds_arr = prefilter_stack(ds, ds_arr, fn_ref_dem, t_vals, filt_ref=filt_ref, ref_dem_date=ref_dem_date,
                                 time_filt_thresh=time_filt_thresh, nproc=nproc)
        print('Elapsed time is ' + str(time.time() - start))

    # constrain variance based on manually defined dependencies
    err_arr, med_slope = constrain_var_slope_corr(ds, ds_arr, ds_corr, t_vals, uncert, fn_ref_dem=fn_ref_dem,
                                                  nproc=nproc)

    if fn_ref_dem is not None:
        ds_arr = robust_wls_ref_filter_stack(ds, ds_arr, err_arr, t_vals, fn_ref_dem,
                                             ref_dem_date=np.datetime64('2013-01-01'),
                                             max_dhdt=time_filt_thresh, nproc=nproc, cutoff_kern_size=1000,
                                             max_deltat_ref=2.,
                                             base_thresh=100.)

    # write variance stats to disk
    if exc_mask is not None:
        print('Deriving variance stats...')
        bins_slope = np.arange(0, 80, 10)
        # fn_stats_slope = os.path.join(os.path.dirname(outfile),os.path.splitext(os.path.basename(outfile))[0]+'_slope_var.csv')
        # get_var_by_bin(ds,ds_arr,med_slope,bins_slope,fn_stats_slope,inc_mask=None,exc_mask=gla_mask,rast_mask_cube=False)

        bins_corr = np.arange(0, 105, 10)
        # fn_stats_corr = os.path.join(os.path.dirname(outfile),
        #                               os.path.splitext(os.path.basename(outfile))[0] + '_corr_var.csv')
        # get_var_by_bin(ds, ds_arr, ds_corr, bins_corr, fn_stats_corr, exc_mask=gla_mask, rast_mask_cube=True)

        fn_stats_both = os.path.join(os.path.dirname(outfile),
                                     os.path.splitext(os.path.basename(outfile))[0] + 'slopecorr_var.csv')
        get_var_by_corr_slope_bins(ds, ds_arr, med_slope, bins_slope, ds_corr, bins_corr, fn_stats_both, inc_mask=None,
                                   exc_mask=exc_mask, nproc=1)
        print('Elapsed time is ' + str(time.time() - start))

        # #TODO: need to estimate dh here first?
        # fn_vgm = os.path.join(os.path.dirname(outfile), os.path.splitext(os.path.basename(outfile))[0] + '_elev_vgm.csv')
        # fn_vgm = os.path.join(os.path.dirname(outfile), os.path.splitext(os.path.basename(outfile))[0] + '_slope_vgm.csv')
        # get_vgm_by_bin(arr_vals, bin_vals, fn_prefilt_stack, outfile, inc_mask=None, exc_mask=None, nproc=1)

    # define temporal prediction output vector
    if tlim is None:
        y0 = t_vals[0].astype('datetime64[D]').astype(object).year
        y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    else:
        y0 = tlim[0].astype('datetime64[D]').astype(object).year
        y1 = tlim[-1].astype('datetime64[D]').astype(object).year
    fit_t = np.arange(y0, y1 + tstep, tstep) - y0
    nice_fit_t = [np.timedelta64(int(d), 'D').astype(int) for d in np.round(fit_t * 365.2524)]

    # converting time values for input vector
    ftime = t_vals
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in ftime])
    time_vals = (ftime_delta / total_delta) * (int(y1) - int(y0))

    print(time_vals)

    print('Fitting with method: ' + method)

    fn_filt = os.path.join(os.path.dirname(outfile), os.path.splitext(os.path.basename(fn_stack))[0] + '_filtered.nc')

    if nproc == 1:
        print('Processing with 1 core...')
        if method == 'gpr':
            out_cube, filt_cube = gpr_wrapper((ds_arr, 0, time_vals, err_arr, fit_t, opt_gpr, kernel, uns_arr))
            cube_to_stack(ds, out_cube, y0, nice_fit_t, slope_arr=med_slope, outfile=outfile, clobber=clobber)
            if write_filt:
                cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True, ci=False)
        elif method in ['ols', 'wls']:
            if method == 'ols':
                weig = False
            else:
                weig = True
            out_arr, filt_cube = ls_wrapper((ds_arr, 0, t_vals, uncert, weig, filt_ls, conf_filt_ls))
            arr_to_img(ds, out_arr, outfile=outfile)
            if write_filt:
                cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True, ci=False)
    else:
        print('Processing with ' + str(nproc) + ' cores...')

        # if not using dask scheduler/chunking, we split manually and process with numba.jit + multiprocessing.pool+map
        if not dask_parallel:
            print('Using multiprocessing...')

            if method in ['ols', 'wls']:
                # here calculation is done matricially so we want to use all cores with the largest tiles possible
                opt_n_tiles = int(np.floor(np.sqrt(nproc)))
                n_x_tiles = opt_n_tiles
                n_y_tiles = opt_n_tiles
            elif method == 'gpr':
                # here calculation is within a for loop: better to have small tiles to get an idea of the processing speed
                n_x_tiles = np.ceil(ds['x'].shape[0] / 30).astype(int)  # break it into 10x10 tiles
                n_y_tiles = np.ceil(ds['y'].shape[0] / 30).astype(int)

                pool = mp.Pool(nproc, maxtasksperchild=1)
                split_arr = splitter(ds_arr, (n_y_tiles, n_x_tiles))
                split_err = splitter(err_arr, (n_y_tiles, n_x_tiles))

                if uns_arr is not None:
                    split_uns = splitter(uns_arr, (n_y_tiles, n_x_tiles))
                else:
                    split_uns = [None] * len(split_arr)

            if method == 'gpr':

                argsin = [(s, i, np.copy(time_vals), split_err[i], np.copy(fit_t), opt_gpr, kernel, split_uns[i]) for
                          i, s in
                          enumerate(split_arr)]
                outputs = pool.map(gpr_wrapper, argsin, chunksize=1)
                pool.close()
                pool.join()

                # this was for when mapped function was giving multiple outputs..
                # zip_out = list(zip(*outputs))
                # out_cube = stitcher(zip_out[0], (n_y_tiles, n_x_tiles))
                # filt_cube = stitcher(zip_out[1], (n_y_tiles, n_x_tiles))

                stitched_outputs = stitcher(outputs, (n_y_tiles, n_x_tiles))

            elif method in ['ols', 'wls']:
                if method == 'ols':
                    weig = False
                else:
                    weig = True
                argsin = [(s, i, np.copy(t_vals), np.copy(uncert), weig, filt_ls, conf_filt_ls) for i, s in
                          enumerate(split_arr)]
                outputs = pool.map(ls_wrapper, argsin, chunksize=1)
                pool.close()
                pool.join()

                zip_out = list(zip(*outputs))

                out_arr = stitcher(zip_out[0], (n_y_tiles, n_x_tiles))
                arr_to_img(ds, out_arr, outfile=outfile)
                if write_filt:
                    filt_cube = stitcher(zip_out[1], (n_y_tiles, n_x_tiles))
                    cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True,
                                  ci=False)

        # here we use dask instead, testing only with gpr for now
        else:
            print('Using dask distributed parallel, computing with chunks sizes of 30...')
            print('Elapsed time is ' + str(time.time() - start))

            if method == 'gpr':

                ds['err'] = (['time', 'y', 'x'], err_arr.astype(np.float32))
                # TODO: look at the details of issue before posting on xarray's GitHub: dem_names dtype "object" not recognized for writing after an "isel"
                ds = ds.drop('dem_names')

                print('Saving data to temporary file...')
                fn_tmp = os.path.join(os.path.dirname(outfile), 'tmp.nc')

                mkdir_p(os.path.dirname(fn_tmp))
                if os.path.exists(fn_tmp):
                    os.remove(fn_tmp)
                ds.to_netcdf(fn_tmp)
                ds.close()

                # TODO: getting an issue similar than this: https://github.com/pydata/xarray/issues/1836, but:
                # persist takes ages to load the data and it happens even without zipping netcdf...

                # ds_dask = xr.open_dataset(fn_tmp).chunk({'x':30,'y':30})
                # ds_dask = ds_dask.persist()

                # we load the data directly to avoid all this trouble
                print('Loading data...')

                ds_dask = xr.open_dataset(fn_tmp)

                z_vals = ds_dask.z.values
                err_vals = ds_dask.err.values

                chunk_z = ds_dask.z.chunk({'x': 150, 'y': 150})
                chunk_err = ds_dask.err.chunk({'x': 150, 'y': 150})

                print('Elapsed time is ' + str(time.time() - start))

                # client = dask.distributed.Client()

                d = gpr_dask(chunk_z, chunk_err, time_vals, fit_t)
                print('Starting compute')
                with ProgressBar():
                    out_ds = d.compute(num_workers=nproc, scheduler='processes')
                out_ds = out_ds.transpose('new_time', 'y', 'x')
                stitched_outputs = out_ds.values

        print('Elapsed time is ' + str(time.time() - start))

        print('Writing results to disk...')

        # now we write results to disk
        out_cube = stitched_outputs[:2 * len(nice_fit_t), :, :]
        cube_to_stack(ds, out_cube, y0, nice_fit_t, slope_arr=med_slope, outfile=outfile, clobber=clobber)
        if write_filt:
            filt_cube = stitched_outputs[2 * len(nice_fit_t):, :, :]
            cube_to_stack(ds, filt_cube, y0, filt_vals, outfile=fn_filt, clobber=clobber, filt_bool=True,
                          ci=False)

        print('Elapsed time is ' + str(time.time() - start))

        # to write back to full extent even if the dataset was subsetted spatially at the top
        # if inc_mask is not None:
        #     full_out_cube = np.zeros((np.shape(out_cube)[0],ds_orig.y.size,ds_orig.x.size)) * np.nan
        #     full_out_cube[slices[0],slices[1]] = out_cube
        #     out_cube = full_out_cube
        #     ds=ds_orig
        # if inc_mask is not None:
        #     full_filt_cube = np.zeros((np.shape(filt_cube)[0], ds_orig.y.size, ds_orig.x.size)) * np.nan
        #     full_filt_cube[slices[0], slices[1]] = filt_cube
        #     filt_cube = full_filt_cube
