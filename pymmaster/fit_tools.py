from __future__ import print_function
import os
import sys
import time
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
import numpy as np
import gdal
import xarray as xr
import matplotlib.pylab as plt
import multiprocessing as mp
from itertools import chain
from scipy import stats
from scipy.interpolate import interp1d
from scipy.ndimage import filters
from skimage.morphology import disk
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, RationalQuadratic as RQ, ExpSineSquared as ESS
from numba import jit
from llc import jit_filter_function
from pybob.GeoImg import GeoImg
from pybob.image_tools import create_mask_from_shapefile
# from pymmaster.stack_tools import create_crs_variable, create_nc
import pymmaster.stack_tools as st
from pybob.ddem_tools import nmad
from warnings import filterwarnings

filterwarnings('ignore')

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

def vgm_1d(t_vals,detrend_elev,lag_cutoff,tstep=0.25):

    # 1D variogram: inspired by http://connor-johnson.com/2014/03/20/simple-kriging-in-python/

    def SVh(P, h, bw):
        # empirical variogram for a single lag
        pd = np.abs(np.subtract.outer(P[:,0], P[:,0]))
        N = pd.shape[0]
        Z = list()
        for i in range(N):
            for j in range(i + 1, N):
                if (pd[i, j] >= h - bw) and (pd[i, j] <= h + bw):
                    Z.append((P[i, 1] - P[j, 1]) ** 2.0)
        if len(Z)>0:
            return np.nansum(Z) / (2.0 * len(Z)), len(Z)
        else:
            return np.nan, 0

    def SV(P, hs, bw):
        # empirical variogram for a collection of lags
        sv = list()
        p = list()
        for h in hs:
            ph, svh = SVh(P, h, bw)
            sv.append(svh)
            p.append(ph)
        sv = [[p[i], sv[i]] for i in range(len(hs))]
        return np.array(sv).T

    ind_valid = ~np.isnan(detrend_elev)
    sample = np.column_stack((t_vals[ind_valid], detrend_elev[ind_valid]))

    hs = np.arange(0, lag_cutoff, tstep)
    sv = SV(sample, hs, tstep)

    return sv

def estimate_vgm(fn_stack,sampmask,nsamp=10000,tstep=0.25,lag_cutoff=None,min_obs=8):

    # estimate 1D variogram for multiple pixels: random sampling within mask

    # load filtered stack
    ds = xr.open_dataset(fn_stack)
    # ds.load()
    ds_arr = ds.variables['z'].values

    # rasterize mask
    mask = get_stack_mask(sampmask, ds)

    # count number of valid temporal observation for each pixel
    nb_arr = np.nansum(~np.isnan(ds_arr),axis=0)
    mask = (mask & np.array(nb_arr>=min_obs))

    # sample a subset
    max_samp = np.count_nonzero(mask)
    index = np.where(mask)
    final_nsamp = min(max_samp,nsamp)
    subset = np.random.choice(max_samp,final_nsamp,replace=False)
    index_subset=(index[0][subset],index[1][subset])
    mask_subset = np.zeros(np.shape(mask),dtype=np.bool)
    mask_subset[index_subset] = True

    ds_samp = ds_arr[:, mask_subset]

    # read and convert time values
    t_vals = ds['time'].values

    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))
    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in t_vals])
    t_scale = (ftime_delta / total_delta) * (int(y1) - y0)

    if lag_cutoff is None:
        lag_cutoff = np.max(t_scale) - np.min(t_scale)

    lags = np.arange(0, lag_cutoff, tstep) + 0.5*tstep

    # get variance/lags and number of pairwise/lags for each pixel
    vdata=np.zeros((len(lags),final_nsamp))
    pdata=np.zeros((len(lags),final_nsamp))

    for i in np.arange(final_nsamp):
        sv = vgm_1d(t_scale, ds_samp[:,i].flatten(),lag_cutoff,tstep=tstep)
        vdata[:,i]=sv[0]
        pdata[:,i]=sv[1]

    ptot = np.nansum(pdata,axis=1)

    # mean variogram accounting for the number of pairwise comparison in each pixel
    vmean = np.nansum(vdata * pdata,axis=1) / ptot
    #'rough' std: between pixels, not accounting for the number of pairwise observation
    vstd = np.nanstd(vdata,axis=1)

    return lags, vmean, vstd

def plot_vgm(lags,vmean,vstd):

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
    # perform independent WLS matricially for optimal processing time

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


def iterative_gpr(time_vals, data_vals, err_vals, time_pred, opt=False, kernel=None):
    if opt:
        optimizer = 'fmin_l_bfgs_b'
        n_restarts_optimizer = 9
    else:
        optimizer = None
        n_restarts_optimizer = 0

    if kernel is None:
        k1 = C(2.0, (1e-2, 1e2)) * RBF(10, (5, 30))  # other kernels to try to add here?
        k2 = C(1.0, (1e-2, 1e2)) * RBF(1, (1, 5))
        k3 = C(10, (1e-3, 1e3)) * RQ(length_scale=30, length_scale_bounds=(30, 1e3))
        kernel = k1 + k2 + k3

        # if we do without training, we don't care about bounds, simplifies the expressions:
        # short seasonality departure with a periodic kernel of 1 year
        # k1 = C(5) * ESS(1,1)

        # long departure from linearity with a RQK
        # k2 = RQ(30)
        # kernel = k1 + k2

    # initializing
    n_out = 1
    niter = 0

    num_finite = data_vals.size
    good_vals = np.isfinite(data_vals)

    while n_out > 0 and num_finite > 2 and niter < 3:
        # if we remove a linear trend, normalize_y should be false...
        gp = GaussianProcessRegressor(kernel=kernel, optimizer=optimizer, n_restarts_optimizer=n_restarts_optimizer,
                                      alpha=err_vals[good_vals], normalize_y=False)
        gp.fit(time_vals[good_vals].reshape(-1, 1), data_vals[good_vals].reshape(-1, 1))
        y_pred, sigma = gp.predict(time_pred.reshape(-1, 1), return_std=True)
        y_, s_ = interp_data(time_pred, y_pred.squeeze(), sigma.squeeze(), time_vals)
        z_score = np.abs(data_vals - y_) / s_
        isin = z_score < 4
        n_out = np.count_nonzero(~isin)

        data_vals[~isin] = np.nan
        time_vals[~isin] = np.nan
        err_vals[~isin] = np.nan

        good_vals = np.isfinite(data_vals)
        num_finite = np.count_nonzero(good_vals)
        niter += 1

    if num_finite <= 2:
        y_pred = np.nan * np.zeros(time_pred.shape)
        sigma = np.nan * np.zeros(time_pred.shape)
        z_score = np.nan * np.zeros(data_vals.shape)
    return y_pred.squeeze(), sigma.squeeze(), z_score, good_vals


# TODO: think we should merge this one with iterative gpr
def gpr(data, t_vals, uncert, opt=False, kernel=None, tstep=0.25):
    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    t_pred = np.arange(y0, y1, tstep) - y0

    # changed to be consistent with new return (not concatenated)
    if np.count_nonzero(np.isfinite(data)) < 2:
        y_pred = np.nan * np.zeros(t_pred.shape)
        sigma = np.nan * np.zeros(t_pred.shape)
        z_score = np.nan * np.zeros(t_pred.shape)
        good_vals = np.isfinite(np.nan * np.zeros(t_pred.size))
        return y_pred, sigma, z_score, good_vals

    ftime = t_vals[np.isfinite(data)]
    fdata = data[np.isfinite(data)]
    ferr = uncert[np.isfinite(data)]
    total_delta = np.datetime64('{}-01-01'.format(int(y1))) - np.datetime64('{}-01-01'.format(int(y0)))

    ftime_delta = np.array([t - np.datetime64('{}-01-01'.format(int(y0))) for t in ftime])
    t_scale = (ftime_delta / total_delta) * (int(y1) - y0)

    try:
        # TODO: need to include the detrending in the iterative gpr fitting/filtering, first linear fit might be heavily biased?
        # try to remove a linear fit from the data before we fit, then add it back in when we're done.
        reg = detrend(t_scale, fdata, ferr)
    except:
        y_pred = np.nan * np.zeros(t_pred.shape)
        sigma = np.nan * np.zeros(t_pred.shape)
        z_score = np.nan * np.zeros(fdata.shape)
        good_vals = np.isfinite(np.nan * np.zeros(fdata.size))
        return y_pred, sigma, z_score, good_vals

    fdata = fdata - reg.predict(t_scale.reshape(-1, 1)).squeeze()
    l_trend = reg.predict(t_pred.reshape(-1, 1)).squeeze()

    # std_nmad_rat = np.std(fdata) / nmad(fdata)
    # if std_nmad_rat > 20:
    #    isout = np.abs(fdata) > 10 * nmad(fdata) 
    # else:
    #    isout = np.abs(fdata) > 4 * np.std(fdata) 

    # fdata = fdata[~isout]
    # t_scale = t_scale[~isout]
    # ferr = ferr[~isout]

    y_pred, sigma, z_score, good_vals = iterative_gpr(t_scale, fdata, ferr, t_pred, opt=opt, kernel=kernel)
    return y_pred + l_trend, sigma, z_score, good_vals


def ls(subarr, t_vals, uncert, weigh, filt_ls=False, conf_filt=0.99):
    T, Y, X = subarr.shape

    z_mat = subarr.reshape(T, Y * X)
    t_mat = np.array([t_vals, ] * Y * X).T
    if weigh:
        w_mat = np.array([1. / uncert ** 2, ] * Y * X).T

    if filt_ls:
        if weigh:
            yl, yu = wls_matrix(t_mat, z_mat, w_mat, conf_interv=conf_filt)[3:5]
        else:
            yl, yu = ols_matrix(t_mat, z_mat, conf_interv=conf_filt)[3:5]

        z_mat[z_mat < yl] = np.nan
        z_mat[z_mat > yu] = np.nan

    if weigh:
        beta1, _, incert_slope = wls_matrix(t_mat, z_mat, w_mat)[0:3]
    else:
        beta1, _, incert_slope = ols_matrix(t_mat, z_mat)[0:3]

    date_min = np.nanmin(t_mat, axis=0)
    date_max = np.nanmax(t_mat, axis=0)
    nb_trend = (~np.isnan(z_mat)).sum(axis=0)

    filter_less_2_DEMs = nb_trend <= 2
    beta1[filter_less_2_DEMs] = np.nan
    incert_slope[filter_less_2_DEMs] = np.nan

    slope = np.reshape(beta1, (Y, X))
    slope_sig = np.reshape(incert_slope, (Y, X))
    nb_dem = np.reshape(nb_trend, (Y, X))
    date_min = np.reshape(date_min, (Y, X))
    date_max = np.reshape(date_max, (Y, X))

    outarr = np.concatenate((slope, slope_sig, nb_dem, date_min, date_max), axis=0)

    return outarr

@jit
def gpr_filter_wrapper(argsin):
    subarr, i, t_vals, uncert, new_t, opt, kernel, tstep = argsin
    start = time.time()
    Y, X = subarr[0].shape
    outarr = np.nan * np.zeros((new_t * 2, Y, X))
    # pixel by pixel
    for x in range(X):
        for y in range(Y):
            tmp_y, tmp_sig = gpr(subarr[:, y, x], t_vals, uncert, opt=opt, kernel=kernel, tstep=tstep)[0:2]
            out = np.concatenate((tmp_y, tmp_sig), axis=0)
            outarr[:, y, x] = out
    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr

@jit
def gpr_fit_wrapper(argsin):
    subarr, i, t_vals, uncert, new_t, opt, kernel, tstep = argsin
    start = time.time()
    Y, X = subarr[0].shape
    outarr = np.nan * np.zeros((new_t * 2, Y, X))
    # pixel by pixel
    for x in range(X):
        for y in range(Y):
            tmp_y, tmp_sig = gpr(subarr[:, y, x], t_vals, uncert, opt=opt, kernel=kernel, tstep=tstep)[0:2]
            out = np.concatenate((tmp_y, tmp_sig), axis=0)
            outarr[:, y, x] = out
    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr

@jit
def ls_filter_wrapper(argsin):
    subarr, i, t_vals, uncert, weigh = argsin
    start = time.time()

    # matrix
    outarr = ls(subarr, t_vals, uncert, weigh)

    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr

@jit
def ls_fit_wrapper(argsin):
    subarr, i, t_vals, uncert, weigh = argsin
    start = time.time()

    # matrix
    outarr = ls(subarr, t_vals, uncert, weigh)

    elaps = time.time() - start
    print('Done with block {}, elapsed time {}.'.format(i, elaps))
    return outarr


def splitter(img, nblocks, overlap=0):
    split1 = np.array_split(img, nblocks[0], axis=1)
    split2 = [np.array_split(im, nblocks[1], axis=2) for im in split1]
    olist = [np.copy(a) for a in list(chain.from_iterable(split2))]
    return olist


def stitcher(outputs, nblocks, overlap=0):
    stitched = []
    if np.array(nblocks).size == 1:
        nblocks = np.array([nblocks, nblocks])
    for i in range(nblocks[0]):
        stitched.append(outputs[i * nblocks[1]])
        for j in range(1, nblocks[1]):
            outind = j + i * nblocks[1]
            stitched[i] = np.concatenate((stitched[i], outputs[outind]), axis=2)
    return np.concatenate(tuple(stitched), axis=1)


def cube_to_stack(ds, ds_arr, y0, nice_fit_t, out_cube, fit_t, outfile, method, clobber=False):
    # TODO: probably a cleaner way to write this without having to call ds + ds_arr, nice_fit_t + y0 + fit_t here to copy shapes...

    if outfile is None:
        outfile = 'fit_' + method + '.nc'

    fit_cube = out_cube[:fit_t.size, :, :]
    sig_cube = out_cube[fit_t.size:, :, :]

    nco, to, xo, yo = st.create_nc(ds_arr[0], outfile=outfile,
                                   clobber=clobber, t0=np.datetime64('{}-01-01'.format(y0)))
    st.create_crs_variable(parse_epsg(ds['crs'].spatial_ref), nco)

    x, y = ds['x'].values, ds['y'].values
    xo[:] = x
    yo[:] = y
    to[:] = nice_fit_t

    zo = nco.createVariable('z', 'f4', ('time', 'y', 'x'), fill_value=-9999)
    zo.units = 'meters'
    zo.long_name = 'Fit elevation above WGS84 ellipsoid'
    zo.grid_mapping = 'crs'
    zo.coordinates = 'x y'
    zo.set_auto_mask(True)

    zo[:] = fit_cube

    fzo = nco.createVariable('z_ci', 'f4', ('time', 'y', 'x'), fill_value=-9999)
    fzo.units = 'meters'
    fzo.long_name = '68% confidence interval for elevation fit.'
    fzo.grid_mapping = 'crs'
    fzo.coordinates = 'x y'
    fzo.set_auto_mask(True)

    fzo[:] = sig_cube

    nco.close()


def arr_to_img(ds, out_arr, outfile, method):
    if outfile is None:
        outfile_fit = 'fit_' + method + '_dh.tif'
        outfile_sig = 'fit_' + method + '_err.tif'
        outfile_nb = 'fit_' + method + '_nb.tif'
        outfile_dmin = 'fit_' + method + '_dmin.tif'
        outfile_dmax = 'fit_' + method + '_dmax.tif'
    else:
        outfile_fit = os.path.join(os.path.dirname(outfile),
                                   os.path.splitext(os.path.basename(outfile))[0] + '_dh.tif')
        outfile_sig = os.path.join(os.path.dirname(outfile),
                                   os.path.splitext(os.path.basename(outfile))[0] + '_err.tif')
        outfile_nb = os.path.join(os.path.dirname(outfile),
                                  os.path.splitext(os.path.basename(outfile))[0] + '_nb.tif')
        outfile_dmin = os.path.join(os.path.dirname(outfile),
                                    os.path.splitext(os.path.basename(outfile))[0] + '_dmin.tif')
        outfile_dmax = os.path.join(os.path.dirname(outfile),
                                    os.path.splitext(os.path.basename(outfile))[0] + '_dmax.tif')

    fit_arr = out_arr[1, :, :]
    sig_arr = out_arr[2, :, :]
    nb_arr = out_arr[3, :, :]
    dmin_arr = out_arr[4, :, :]
    dmax_arr = out_arr[5, :, :]

    # TODO: I don't get how you go from stack back to GeoImg easily yet
    # arr = geoimg(ds)

    # arr.img = fit_arr
    # arr.write(outfile_fit)
    # arr.img = sig_arr
    # arr.write(outfile_sig)
    # arr.img = nb_arr
    # arr.write(outfile_nb)
    # arr.img = dmin_arr
    # arr.write(outfile_dmin)
    # arr.img = dmax_arr
    # arr.write(outfile_dmax)


def time_filter_ref(ds, t_vals, ref_dem, ref_date, thresh=50, base_thresh=20):
    delta_t = (ref_date - t_vals).astype('timedelta64[D]').astype(float) / 365.24
    dh = ds - ref_dem.img
    dt_arr = np.ones(dh.shape)
    for i, d in enumerate(delta_t):
        dt_arr[i] = dt_arr[i] * d
    if np.array(thresh).size == 1:
        ds[np.abs(dh) > base_thresh + np.abs(dt_arr)*thresh] = np.nan
    else:
        #TODO: not sure about how I wrote this one... need to define the signs of thresh[0] and thresh[1]
        ds[np.logical_and(dh > base_thresh + dt_arr*thresh[0], dh < -base_thresh + dt_arr*thresh[1])] = np.nan
    #     ds[np.logical_and(d_data < thresh[0],
    #                       d_data > thresh[1])] = np.nan
    return ds


def spat_filter_ref(ds_arr, ref_dem, cutoff_kern_size=5000, cutoff_thr=100.):

    @jit_filter_function
    def nanmax(a):
        return np.nanmax(a)

    @jit_filter_function
    def nanmin(a):
        return np.nanmin(a)
    # here we assume that the reference DEM is a "clean" post-processed DEM, filtered with QA for low confidence outliers
    # minimum/maximum elevation in circular surroundings based on reference DEM

    ref_arr = ref_dem.img
    res = ref_dem.dx

    rad = int(np.floor(cutoff_kern_size / res))
    max_arr = filters.generic_filter(ref_arr, nanmax, footprint=disk(rad))
    min_arr = filters.generic_filter(ref_arr, nanmin, footprint=disk(rad))

    for i in range(ds_arr.shape[0]):
        ds_arr[i, np.logical_or(ds_arr[i, :] > (max_arr + cutoff_thr), ds_arr[i, :] < (min_arr - cutoff_thr))] = np.nan

    return ds_arr

def fit_stack(fn_stack, fn_ref_dem=None, ref_dem_date=None, filt='min_max', filter_thresh=50, inc_mask=None, nproc=1, method='gpr', opt_gpr=False,
              kernel=None, cutoff_trange=None, tstep=0.25, outfile='fit.nc', clobber=False, return_fit=True):
    """
    Given a netcdf stack of DEMs, perform temporal fitting with uncertainty propagation

    :param fn_stack: Filename for input netcdf file
    :param fn_ref_dem: Filename for input reference DEM (maybe we change that to a footprint shapefile to respect your original structure?)
    :param ref_dem_date: Date of ref_dem
    :param filt: Type of filtering
    :param filter_thresh: Maximum dh/dt from reference DEM for time filtering
    :param inc_mask: Optional inclusion mask
    :param nproc: Number of cores for multiprocessing [1]
    :param method: Fitting method, currently supported: Gaussian Process Regression "gpr", Ordinary Least Squares "ols" and Weighted Least Squares "wls" ["gpr"]
    :param opt_gpr: Run learning optimization in the GPR fitting [False]
    :param kernel: Kernel
    :param cutoff_trange: Temporal range to fit
    :param tstep: Temporal step for fitted stack [0.25 year]
    :param outfile: Path to outfile
    :return:
    """

    assert method in ['gpr', 'ols', 'wls'], "Method must be one of gpr, ols or wls."
    print('Reading dataset: ' + fn_stack)

    # we are already chunking manually (split/stitch) below so let's leave chunking aside for now
    # ds = xr.open_dataset(fn_stack, chunks={'x': 100, 'y': 100})

    ds = xr.open_dataset(fn_stack)
    # ds.load()
    ds_arr = ds.variables['z'].values

    if inc_mask is not None:
        land_mask = get_stack_mask(inc_mask, ds)
        ds_arr[:, ~land_mask] = np.nan

    print('Filtering...')
    keep_vals = ds['uncert'].values < 20
    t_vals = ds['time'].values[keep_vals]
    uncert = ds['uncert'].values[keep_vals]
    ds_arr = ds_arr[keep_vals, :, :]

    # minimum/maximum elevation on Earth
    ds_arr[np.logical_or(ds_arr < -400, ds_arr > 8900)] = np.nan

    if fn_ref_dem is not None:
        assert filt in ['min_max', 'time','both'], "fn_ref must be one of: min_max, time, both"
        tmp_geo = st.make_geoimg(ds)
        tmp_dem = GeoImg(fn_ref_dem)
        ref_dem = tmp_dem.reproject(tmp_geo)
        if filter == 'min_max':
            print('Filtering using min/max values in {}'.format(fn_ref_dem))
            ds_arr = spat_filter_ref(ds_arr, ref_dem)
        elif filter == 'time':
            if ref_dem_date is None:
                print('Reference DEM time stamp not specified, defaulting to 01.01.2000')
                ref_dem_date = np.datetime64('2000-01-01')
            print('Filtering using dh/dt value to reference DEM, threshold of {} m/a'.format(filter_thresh))
            ds_arr = time_filter_ref(ds_arr, t_vals, ref_dem, ref_dem_date, thresh=filter_thresh)
        elif filter == 'both':
            print('Filtering using min/max values in {}'.format(fn_ref_dem))
            ds_arr = spat_filter_ref(ds_arr, ref_dem)
            if ref_dem_date is None:
                print('Reference DEM time stamp not specified, defaulting to 01.01.2000')
                ref_dem_date = np.datetime64('2000-01-01')
            print('Filtering using dh/dt value to reference DEM, threshold of {} m/a'.format(filter_thresh))
            ds_arr = time_filter_ref(ds_arr, t_vals, ref_dem, ref_dem_date, thresh=filter_thresh)

    y0 = t_vals[0].astype('datetime64[D]').astype(object).year
    y1 = t_vals[-1].astype('datetime64[D]').astype(object).year + 1.1
    fit_t = np.arange(y0, y1, 0.25) - y0
    nice_fit_t = [np.timedelta64(int(d), 'D').astype(int) for d in np.round(fit_t * 365.2524)]

    print('Fitting with method: ' + method)

    if nproc == 1:
        print('Processing with 1 core...')
        if method == 'gpr':
            out_cube = gpr_fit_wrapper((ds_arr, 0, t_vals, uncert, fit_t.size), opt_gpr, kernel, tstep)
            cube_to_stack(ds, ds_arr, y0, nice_fit_t, out_cube, fit_t, outfile=outfile, method=method, clobber=clobber)
        elif method in ['ols', 'wls']:
            if method == 'ols':
                weig = False
            else:
                weig = True
            out_arr = ls_fit_wrapper((ds_arr, 0, t_vals, uncert, fit_t.size), weig)
            arr_to_img(ds, out_arr, outfile=outfile, method=method)
    else:
        print('Processing with ' + str(nproc) + ' cores...')
        # now, try to figure out the nicest way to break up the image, given the number of processors
        # there has to be a better way than this... i'll look into it

        # n_x_tiles = np.ceil(ds['x'].shape[0] / 10).astype(int)  # break it into 10x10 tiles
        # n_y_tiles = np.ceil(ds['y'].shape[0] / 10).astype(int)
        n_x_tiles = 8
        n_y_tiles = 8
        pool = mp.Pool(nproc, maxtasksperchild=1)
        split_arr = splitter(ds_arr, (n_y_tiles, n_x_tiles))

        if method == 'gpr':
            argsin = [(s, i, np.copy(t_vals), np.copy(uncert), np.copy(fit_t.size), opt_gpr, kernel, tstep) for i, s in
                      enumerate(split_arr)]
            outputs = pool.map(gpr_fit_wrapper, argsin, chunksize=1)
            out_cube = stitcher(outputs, (n_y_tiles, n_x_tiles))
            pool.close()
            pool.join()

            cube_to_stack(ds, ds_arr, y0, nice_fit_t, out_cube, fit_t, outfile=outfile, method=method, clobber=clobber)

        elif method in ['ols', 'wls']:
            if method == 'ols':
                weig = False
            else:
                weig = True
            argsin = [(s, i, np.copy(t_vals), np.copy(uncert), weig) for i, s in
                      enumerate(split_arr)]
            outputs = pool.map(ls_fit_wrapper, argsin, chunksize=1)
            out_arr = stitcher(outputs, (n_y_tiles, n_x_tiles))
            pool.close()
            pool.join()

            arr_to_img(ds, out_arr, outfile=outfile, method=method)