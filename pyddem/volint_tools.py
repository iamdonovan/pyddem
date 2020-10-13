"""
pymmaster.volint_tools provides tools to integrate elevation change into volume time series, adapted from McNabb et al. (2019)
"""
import sys
import numpy as np
import pandas as pd
import math as m
import pyddem.fit_tools as ft
import functools
from pyddem.spstats_tools import neff_rect, neff_circ, kernel_sph, std_err_finite, std_err

def idx_near_val(array, v):
    return np.nanargmin(np.abs(array - v))

def linear_err(delta_x, std_acc_y):
    return delta_x ** 2 / 8 * std_acc_y


def gauss(n=11, sigma=1):
    r = range(-int(n / 2), int(n / 2) + 1)
    return [1 / (sigma * m.sqrt(2 * m.pi)) * m.exp(-float(x) ** 2 / (2 * sigma ** 2)) for x in r]


def kernel_exclass(xi, x0, a1, kappa=0.5):
    return np.exp(-(np.abs(xi - x0) / a1) ** kappa)


def kernel_exp(xi, x0, a1):
    return np.exp(-np.abs(xi - x0) / a1)


def kernel_gaussian(xi, x0, a1):
    return np.exp(-((xi - x0) / a1) ** 2)


def kernel_sph(xi,x0,a1):
    if np.abs(xi - x0) > a1:
        return 0
    else:
        return 1 - 3 / 2 * np.abs(xi-x0) / a1 + 1 / 2 * (np.abs(xi-x0) / a1) ** 3


def lowess_homemade_kern(x, y, w, a1, kernel='Exp'):
    """
    #inspired by: https://xavierbourretsicotte.github.io/loess.html
    homebaked lowess with variogram kernel + heteroscedasticity of observations with error

    :param x:
    :param y:
    :param w: heteroscedastic weights (inverse of variance)
    :param a1: range of the kernel (in variogram terms)
    :param kernel: kernel function
    :return:
    """

    n = len(x)
    yest = np.zeros(n)
    err_yest = np.zeros(n)

    if kernel == 'Gau':
        kernel_fun = kernel_gaussian
    elif kernel == 'Exp':
        kernel_fun = kernel_exp
    elif kernel == 'Exc':
        kernel_fun = kernel_exclass
    else:
        print('Kernel not recognized.')
        sys.exit()

    W = np.array([kernel_fun(x, x[i], a1) * w for i in range(n)]).T
    X = np.array([x for i in range(n)]).T
    Y = np.array([y for i in range(n)]).T

    beta1, beta0, _, Yl, Yu = ft.wls_matrix(X, Y, W, conf_interv=0.68)

    for i in range(n):
        yest[i] = beta1[i] * x[i] + beta0[i]
        err_yest[i] = (Yu[i, i] - Yl[i, i]) / 2

    return yest, err_yest


def interp_linear(xp, yp, errp, acc_y, loo=False):
    # interpolation 1d with error propagation: nan are considered void, possible leave-one-out (reinterpolate each value)

    # getting void index
    idx_void = np.isnan(yp)

    # preallocating arrays
    yp_out = np.copy(yp)
    errp_out = np.copy(errp)
    errlin_out = np.zeros(len(yp)) * np.nan

    # don't really care about performance, it's fast anyway, so let's do this one at a time
    for i in np.arange(len(xp)):

        x0 = xp[i]
        tmp_xp = np.copy(xp)
        tmp_xp[idx_void] = np.nan
        if loo:
            tmp_xp[i] = np.nan  # this is for leave-one out
        else:
            if not np.isnan(tmp_xp[i]):
                continue

        # find closest non void bin
        idx_1 = idx_near_val(tmp_xp, x0)
        tmp_xp[idx_1] = np.nan
        # second closest
        idx_2 = idx_near_val(tmp_xp, x0)

        # linear interpolation (or extrapolation)
        a = (xp[idx_2] - x0) / (xp[idx_2] - xp[idx_1])
        b = (x0 - xp[idx_1]) / (xp[idx_2] - xp[idx_1])

        # propagating standard error
        y0_out = a * yp[idx_1] + b * yp[idx_2]
        err0_out = np.sqrt(a ** 2 * errp[idx_1] ** 2 + b ** 2 * errp[idx_2] ** 2)
        # err0_out = np.sqrt(errp[idx_1]**2 + errp[idx_2]**2)

        # estimating linear error
        delta_x = min(np.absolute(xp[idx_2] - x0), np.absolute(xp[idx_1] - x0))
        errlin0_out = linear_err(delta_x, acc_y)

        # appending
        yp_out[i] = y0_out
        errp_out[i] = err0_out
        errlin_out[i] = errlin0_out

    return yp_out, errp_out, errlin_out


def interp_lowess(xp, yp, errp, acc_y, rang, kernel='Exc'):
    # interp1d with local regression and error propagation

    yp_out, errp_out = lowess_homemade_kern(xp, yp, 1 / (errp ** 2), a1=rang / 4., kernel=kernel)

    idx_void = np.isnan(yp)
    errlin_out = np.zeros(len(yp)) * np.nan

    for i in np.arange(len(xp)):

        x0 = xp[i]
        tmp_xp = np.copy(xp)
        tmp_xp[idx_void] = np.nan
        if not np.isnan(tmp_xp[i]):
            continue

        # find closest non void bin
        idx_1 = idx_near_val(tmp_xp, x0)
        tmp_xp[idx_1] = np.nan

        delta_x = np.absolute(xp[idx_1] - x0)
        errlin0_out = linear_err(delta_x, acc_y)
        errlin_out[i] = np.sqrt(errlin0_out ** 2 + errp[idx_1] ** 2)

    return yp_out, errp_out, errlin_out

def double_sum_covar_hypso(tot_err, slope_bin, elev_bin, area_tot, crange, kernel):
    n = len(tot_err)

    dist_bin = np.zeros(n)
    # change elev binning in distances:
    bin_size = elev_bin[1] - elev_bin[0]

    for i in range(n - 1):
        ind = elev_bin <= elev_bin[i]
        tmpslope = slope_bin[ind]

        dist_bin[i + 1] = np.nansum(bin_size / np.tan(tmpslope * np.pi / 180.))

    var_err = 0
    for i in range(n):
        for j in range(n):
            var_err += kernel(dist_bin[i], dist_bin[j], crange) * tot_err[i] * tot_err[j] * area_tot[i] * area_tot[j]

    var_err /= np.nansum(area_tot) ** 2

    return np.sqrt(var_err)

def hypso_dc(dh_dc, err_dc, ref_elev, dt, tvals, mask, gsd, slope=None, bin_type='fixed', bin_val=100., filt_bin='5NMAD', method='linear'):


    #filtering with error threshold?
    filt_err = err_dc[-1, :] > 500.
    dh_dc[:,filt_err] = np.nan
    err_dc[:,filt_err] = np.nan
    # ref_elev[filt_err] = np.nan

    area_tot = np.count_nonzero(mask) * gsd ** 2

    #valid points for volume change calculation
    valid_points = np.count_nonzero(np.logical_and.reduce((~np.isnan(ref_elev),~np.isnan(dh_dc[0,:]),mask)))

    #closest observation binning: monthly
    dt_bin = np.arange(0,np.nanmax(dt)+30,5)
    nb_dt_bin = len(dt_bin)-1

    #amplitude of correlation length, calibrated with ICESat global
    corr_ranges = [150,2000,5000,20000,50000]

    coefs=[np.array([1.26694247e-03, 3.03486839e+00]),
        np.array([1.35708936e-03, 4.05065698e+00]),
        np.array([1.42572733e-03, 4.20851582e+00]),
        np.array([1.82537137e-03, 4.28515920e+00]),
        np.array([1.87250755e-03, 4.31311254e+00]),
        np.array([2.06249620e-03, 4.33582812e+00])]
    thresh = [0,0,0,180,180]
    ind = [1,1,1,2,1]
    # corr_a = [30,35,35]
    # corr_b = [0.2,0.02,0.0002]
    # corr_c = [0.25,0.5,0.85]
    # lin_a = [35]
    # lin_b = [0.005]
    # def sill_frac(t,a,b,c,d):
    #     # ind_pos = (t-d) >= 0
    #     # out_frac = np.zeros(len(t))
    #     # out_frac[ind_pos] = a*(1 - np.exp(-b*(t[ind_pos]-d)**c))
    #     # return out_frac
    #     if t > d:
    #         return a*(1 - np.exp(-b*(t-d)**c))
    #     else:
    #         return 0
    def sill_frac(t,a,b,c,d):
        if t>=c:
            return (coefs[-1][0]*t+coefs[-1][1])**2-(a*t+b)**2 -((coefs[-1][1]+c*coefs[-1][0])**2 - (coefs[-1-d][1]+c*coefs[-1-d][0])**2)
        else:
            return 0

    corr_std_dt = [functools.partial(sill_frac,a=coefs[i][0],b=coefs[i][1],c=thresh[i],d=ind[i]) for i in range(len(corr_ranges))]
    # corr_std_dt.append(functools.partial(sill_frac_lin,b=lin_b[0]))

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # vec = np.arange(0,1500,5)
    # for i in range(5):
    #     plt.plot(vec,[corr_std_dt[i](j) for j in vec],label=corr_ranges[i])

    if valid_points == 0:
        df = pd.DataFrame()
        df = df.assign(hypso=np.nan, time=np.nan, dh=np.nan, err_dh=np.nan)
        df_hyp = pd.DataFrame()
        df_hyp = df_hyp.assign(hypso=np.nan, area_meas=np.nan, area_tot=np.nan, nmad=np.nan)
        df_int = pd.DataFrame()
        df_int = df_int.assign(time=tvals, dh=np.nan, err_dh=np.nan)

        return df, df_hyp, df_int

    #elevation binning
    min_elev = np.nanmin(ref_elev[mask]) - (np.nanmin(ref_elev[mask]) % bin_val)
    max_elev = np.nanmax(ref_elev[mask])

    if max_elev == min_elev:
        max_elev = min_elev + 0.01
    if bin_type == 'fixed':
        bin_final = bin_val
    elif bin_type == 'percentage':
        bin_final = np.ceil(bin_val / 100. * (max_elev - min_elev))
    else:
        sys.exit('Bin type not recognized.')

    if valid_points > 50 :
        bins_on_mask = np.arange(min_elev, max_elev, bin_final)
        nb_bin = len(bins_on_mask)
    else:
        bins_on_mask = np.array([min_elev])
        bin_final = max_elev + 0.01 - min_elev
        nb_bin = 1

    #index only glacier pixels
    ref_on_mask = ref_elev[mask]
    dh_on_mask = dh_dc[:, mask]
    err_on_mask = err_dc[:, mask]
    dt_on_mask=dt[:,mask]

    #local hypsometric method (McNabb et al., 2019)

    #preallocating
    elev_bin, slope_bin, area_tot_bin, area_res_bin, area_meas_bin, nmad_bin, std_geo_err = (np.zeros(nb_bin) * np.nan for i in range(7))
    mean_bin, std_bin, ss_err_bin, std_num_err, std_all_err = (np.zeros((nb_bin, np.shape(dh_dc)[0])) * np.nan for i in range(5))
    final_num_err_corr, int_err_corr = (np.zeros((np.shape(dh_dc)[0],len(corr_ranges)+1))*np.nan for i in range(2))

    for i in np.arange(nb_bin):

        idx_bin = np.array(ref_on_mask >= bins_on_mask[i]) & np.array(
            ref_on_mask < (bins_on_mask[i] + bin_final))
        idx_orig = np.array(ref_elev >= bins_on_mask[i]) & np.array(
            ref_elev < (bins_on_mask[i] + bin_final)) & mask
        area_tot_bin[i] = np.count_nonzero(idx_orig) * gsd ** 2
        elev_bin[i] = bins_on_mask[i] + bin_final / 2.
        if slope is None:
            slope_bin[i] = 30. #average slope
        else:
            slope_bin[i] = np.nanmean(slope[idx_orig])

        dh_bin = dh_on_mask[:, idx_bin]
        err_bin = err_on_mask[:, idx_bin]

        # with current fit, a value can only be NaN along the entire temporal axis
        nvalid = np.count_nonzero(~np.isnan(dh_bin[0, :]))
        area_res_bin[i] = nvalid * gsd **2
        area_meas_bin[i] = nvalid * gsd ** 2

        if nvalid > 0:

            if filt_bin == '5NMAD' and nvalid > 10:

                # this is for a temporally varying NMAD, which doesn't seem to always be relevant...
                # need to change preallocation if using this one
                #     mad = np.nanmedian(np.absolute(dh_bin - med_bin[i, :, None]), axis=1)
                #     nmad_bin[i, :] = 1.4826 * mad
                #     idx_outlier = np.absolute(dh_bin - med_bin[i, :, None]) > 3 * nmad_bin[i, :, None]
                #     dh_bin[idx_outlier] = np.nan

                # this is for a fixed NMAD using only the max dh

                dh_tot = dh_bin[-1,:]
                med_tot = np.nanmedian(dh_tot)
                mad = np.nanmedian(np.absolute(dh_tot - med_tot))

                nmad_bin[i] = 1.4826*mad
                idx_outlier = np.absolute(dh_tot - med_tot) > 5 * nmad_bin[i]
                nb_outlier = np.count_nonzero(idx_outlier)
                dh_bin[:,idx_outlier] = np.nan
                err_bin[:,idx_outlier] = np.nan
                area_meas_bin[i] -= nb_outlier * gsd ** 2

                # ref_elev_out[idx_orig & np.array(np.absolute(ref_elev_out - med_bin[i]) > 3 * nmad)] = np.nan

            std_bin[i, :] = np.nanstd(dh_bin, axis=1)

            #normal mean
            # mean_bin[i,:] = np.nanmean(dh_bin,axis=1)

            # weighted mean
            weights = 1. / err_bin ** 2
            mean_bin[i, :] = np.nansum(dh_bin * weights, axis=1) / np.nansum(weights, axis=1)
            # print(err_bin)
            # print(weights)
            # print(mean_bin[i,:])
            ss_err_bin[i, :] = np.sqrt(np.nansum(err_bin **2 * weights, axis=1) / np.nansum(weights, axis=1))

            # ref_elev_out[idx_orig & np.isnan(ref_elev_out)] = mean_bin[i]

    # area_tot = np.nansum(area_tot_bin)
    area_meas = np.nansum(area_meas_bin)
    perc_area_res = np.nansum(area_res_bin)/area_tot
    perc_area_meas = area_meas/area_tot
    idx_nonvoid = area_meas_bin > 0

    #what kind of signal are we missing? this is dependent on the hypsometric elevation change correlation
    # over void areas, i.e. the size of the whole glacier
    crange_geo = 10**(2.+area_tot/10**6/2.5) #in meters, this is a decent empirical approximation of exponential hypsometric correlation length based on area size

    # what is the typical std of glacier elevation change: proportional to dh
    psill_bin = std_bin[:, -1]
    # psill_bin = mean_bin[:, -1]

    for i in range(len(area_tot_bin)):
        if idx_nonvoid[i]:
            Neff_geo = neff_rect(area_tot_bin[i],bin_final/(np.tan(slope_bin[i]*np.pi/180.)),crange_geo,psill_bin[i],model1='Exp')
            neff_geo = neff_rect(area_meas_bin[i],bin_final/(np.tan(slope_bin[i]*np.pi/180.)),crange_geo,psill_bin[i],model1='Exp')
            std_geo_err[i] = std_err_finite(psill_bin[i],Neff_geo,neff_geo)
        else:
            std_geo_err[i] = np.nan

    #what is our numerical error
    crange_num1 = 100 #ASTER base correlation length
    psill_num1 = 4**2 #ASTER base variance
    crange_num2 = 1500 #ASTER jitter correlation length
    psill_num2 = 0.6**2 #ASTER jitter has a mean amplitude of ~2m, let's say we corrected or filtered about 70%

    # first, get standard error for all non-void bins
    for i in range(len(area_tot_bin)):
        if idx_nonvoid[i]:
            neff = neff_rect(area_meas_bin[i],bin_final/(np.tan(slope_bin[i]*np.pi/180.)),crange1=crange_num1,psill1=psill_num1
                             ,model1='Sph',crange2=crange_num2,psill2=psill_num2,model2='Sph')
            std_num_err[i,:] = std_err(ss_err_bin[i,:], neff)
        else:
            std_num_err[i,:] = np.nan

    std_num_err[np.isnan(std_num_err)] = 0
    std_geo_err[np.isnan(std_geo_err)] = 0

    std_all_err[idx_nonvoid] = np.sqrt(std_num_err[idx_nonvoid] ** 2 + std_geo_err[idx_nonvoid,None] ** 2)

    # if method == 'linear':

        # # first, do a leave-one out linear interpolation to remove non-void bins with really low confidence
        # loo_mean, loo_std_err, loo_lin_err = interp_linear(elev_bin, mean_bin, std_all_err, acc_dh, loo=True)
        # loo_full_err = np.sqrt(loo_std_err ** 2 + loo_lin_err ** 2)
        #
        # idx_low_conf = nonvoid_err_bin > loo_full_err
        #
        # idx_final_void = np.logical_and(np.invert(idx_nonvoid), idx_low_conf)

        # then, interpolate for all of those bins
        # mean_bin[idx_final_void] = np.nan
        # nonvoid_err_bin[idx_final_void] = np.nan

        # final_std_err[~idx_final_void] = 0

    #linear interpolation of voids by temporal step
    # #TODO: optimize that temporally later

    final_mean_hyp, final_err_hyp = (np.zeros(np.shape(mean_bin))*np.nan for i in range(2))
    final_mean, final_err = (np.zeros(np.shape(mean_bin)[1])*np.nan for i in range(2))

    # without taking into account large scale spatial correlation
    # neff_num_tot = neff_circ(area_meas,crange1=crange_num1,psill1=psill_num1
    #                          ,model1='Sph',crange2=crange_num2,psill2=psill_num2,model2='Sph')

    intrabin_err = std_geo_err
    intrabin_err[np.isnan(intrabin_err)] = 0

    nvalid_bin = np.count_nonzero(idx_nonvoid)
    if nb_bin>1:
        final_geo_err = double_sum_covar_hypso(intrabin_err, slope_bin, elev_bin, area_tot_bin, crange_geo, kernel=kernel_exp)
    else:
        final_geo_err = intrabin_err[0]
    for i in range(np.shape(mean_bin)[1]):
        if nb_bin>1 and nvalid_bin>1:
            tmp_mean, tmp_std_err, tmp_lin_err = interp_linear(elev_bin, mean_bin[:,i], std_all_err[:,i], acc_y=.002, loo=False)
        elif nb_bin>1 and nvalid_bin==1:
            tmp_mean = mean_bin[:,i]
            tmp_std_err = std_all_err[:,i]
            tmp_lin_err = np.zeros(np.shape(mean_bin[:,i]))
        else:
            tmp_mean = np.array([mean_bin[0,i]])
            tmp_std_err = np.array([std_all_err[0,i]])
            tmp_lin_err = np.array([0])

        tmp_std_err[np.isnan(tmp_std_err)] = 0
        tmp_lin_err[np.isnan(tmp_lin_err)] = 0
        interbin_err = np.sqrt(tmp_std_err ** 2 + tmp_lin_err ** 2)
        tmp_tot_err = np.sqrt(interbin_err ** 2 + intrabin_err ** 2)

        final_mean_hyp[:, i] = tmp_mean
        final_err_hyp[:, i] = tmp_tot_err

        # integrate along hypsometry
        final_mean[i] = np.nansum(tmp_mean * area_tot_bin)/area_tot

        #without taking account a very large spatial correlation at the regional scale, only one error:
        # final_num_err = std_err(np.nansum(ss_err_bin[:,i]*area_meas_bin)/area_meas,neff_num_tot)


        nsamp_dt = np.zeros(nb_dt_bin)*np.nan
        err_corr = np.zeros((nb_dt_bin,len(corr_ranges)+1)) * np.nan

        for j in np.arange(nb_dt_bin):

            idx_dt_bin= np.logical_and(dt_on_mask[i,:] >= dt_bin[j],dt_on_mask[i,:]<dt_bin[j+1])

            err_dt = err_on_mask[i,idx_dt_bin]

            final_num_err_dt=np.sqrt(np.nanmean(err_dt**2))
            nsamp_dt[j] = np.count_nonzero(idx_dt_bin)

            sum_var = 0
            for k in range(len(corr_ranges)+1):

                if k != len(corr_ranges):
                    err_corr[j, k] = np.sqrt(max(0,corr_std_dt[len(corr_ranges)-1-k](dt_bin[j] + (dt_bin[j + 1] - dt_bin[j]) / 2) - sum_var))
                    sum_var += err_corr[j, k] ** 2
                else:
                    err_corr[j, k]=np.sqrt(max(0,final_num_err_dt**2-sum_var))


        for k in range(len(corr_ranges)+1):
            final_num_err_corr[i,k] = np.sqrt(np.nansum(err_corr[:,k]*nsamp_dt)/np.nansum(nsamp_dt))

            if k==0:
                tmp_length = 500000
            else:
                tmp_length = corr_ranges[len(corr_ranges)-k]

            if final_num_err_corr[i,k] ==0:
                int_err_corr[i,k] = 0
            else:
                int_err_corr[i,k] = std_err(final_num_err_corr[i,k],neff_circ(area_meas,[(tmp_length,'Sph',final_num_err_corr[i,k]**2)]))

        #TODO: could rewrite this to do it as sum of previous variances

        list_vgm = [(corr_ranges[l],'Sph',final_num_err_corr[i,len(corr_ranges)-l]**2) for l in range(len(corr_ranges))] + [(200000,'Sph',final_num_err_corr[i,0]**2)]

        # list_vgm = [(corr_ranges[0],'Sph',final_num_err_corr[i,3]**2),(corr_ranges[1],'Sph',final_num_err_corr[i,2]**2),
        #             (corr_ranges[2],'Sph',final_num_err_corr[i,1]**2),(500000,'Sph',final_num_err_corr[i,0]**2)]
        neff_num_tot = neff_circ(area_meas,list_vgm)

        final_num_err = std_err(np.sqrt(np.nansum(final_num_err_corr[i,:]**2)),neff_num_tot)

        final_err[i] = np.sqrt(final_num_err**2+final_geo_err**2)

    mean_dt = np.nanmean(dt_on_mask,axis=1)
    std_dt = np.nanstd(dt_on_mask,axis=1)

    #prepare index to write results
    hypso_index = np.array([h for h in elev_bin for t in tvals])
    time_index = np.array([t for h in elev_bin for t in tvals])

    #dataframe with hypsometric mean and error for all time steps
    df = pd.DataFrame()
    df = df.assign(hypso=hypso_index, time=time_index, dh=final_mean_hyp.flatten(), err_dh=final_err_hyp.flatten())

    #dataframe with hypsometric data
    df_hyp = pd.DataFrame()
    df_hyp = df_hyp.assign(hypso=elev_bin,area_meas=area_meas_bin,area_tot=area_tot_bin,nmad=nmad_bin)

    #dataframe with spatially integrated volume for all time steps
    df_int = pd.DataFrame()
    df_int = df_int.assign(time=tvals,dh=final_mean,err_dh=final_err,dt=mean_dt,std_dt=std_dt,
                           perc_area_meas = np.repeat(perc_area_meas,len(tvals)), perc_area_res=np.repeat(perc_area_res,len(tvals)))
    for i in range(len(corr_ranges)):
        df_int['err_corr_'+str(corr_ranges[i])] =int_err_corr[:,len(corr_ranges)-i]
    df_int['err_corr_200000'] = int_err_corr[:,0]
        # err_corr05 = int_err_corr[:, 3], err_corr5 = int_err_corr[:, 2],
        # err_corr50 = int_err_corr[:, 1], err_corr500 = int_err_corr[:, 0],

    # for i in np.arange(nb_bin):
    #     idx_orig = np.array(ref_elev >= bins_on_mask[i]) & np.array(
    #         ref_elev < (bins_on_mask[i] + bin_final)) & mask
    #     if not idx_nonvoid[i]:
    #         ref_elev_out[idx_orig] = final_mean[i]

    # return df, dc_out

    return df, df_hyp, df_int

