"""
pyddem.spstats_tools provides tools to derive spatial/temporal statistics for elevation change data.
"""
from __future__ import print_function
import os
import numpy as np
import math as m
import skgstat as skg
from scipy import integrate
import multiprocessing as mp
import pandas as pd
import ogr
from pyddem.vector_tools import coord_trans, latlon_to_UTM

def neff_sphsum_circular(area,crange1,psill1,crange2,psill2):

    #short range variogram
    c1 = psill1 # partial sill
    a1 = crange1  # short correlation range

    #long range variogram
    c1_2 = psill2
    a1_2 = crange2 # long correlation range

    h_equiv = np.sqrt(area / np.pi)

    #hypothesis of a circular shape to integrate variogram model
    if h_equiv > a1_2:
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * a1_2 ** 2 / (5 * h_equiv ** 2))
    elif (h_equiv < a1_2) and (h_equiv > a1):
        std_err = np.sqrt(c1 * a1 ** 2 / (5 * h_equiv ** 2) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))
    else:
        std_err = np.sqrt(c1 * (1-h_equiv / a1+1 / 5 * (h_equiv / a1) ** 3) + c1_2 * (1-h_equiv / a1_2+1 / 5 * (h_equiv / a1_2) ** 3))

    return (psill1 + psill2)/std_err**2

def neff_circ(area,list_vgm):

    psill_tot = 0
    for vario in list_vgm:
        psill_tot += vario[2]

    def hcov_sum(h):
        fn = 0
        for vario in list_vgm:
            crange, model, psill = vario
            fn += h*(cov(h,crange,model=model,psill=psill))

        return fn

    h_equiv = np.sqrt(area / np.pi)

    full_int = integrate_fun(hcov_sum,0,h_equiv)[0]
    std_err = np.sqrt(2*np.pi*full_int / area)

    return psill_tot/std_err**2


def neff_rect(area,width,crange1,psill1,model1='Sph',crange2=None,psill2=None,model2=None):

    def hcov_sum(h,crange1=crange1,psill1=psill1,model1=model1,crange2=crange2,psill2=psill2,model2=model2):

        if crange2 is None or psill2 is None or model2 is None:
            return h*(cov(h,crange1,model=model1,psill=psill1))
        else:
            return h*(cov(h,crange1,model=model1,psill=psill1)+cov(h,crange2,model=model2,psill=psill2))

    width = min(width,area/width)

    full_int = integrate_fun(hcov_sum,0,width/2)[0]
    bin_int = np.linspace(width/2,area/width,100)
    for i in range(len(bin_int)-1):
        low = bin_int[i]
        upp = bin_int[i+1]
        mid = bin_int[i] + (bin_int[i+1]- bin_int[i])/2
        piec_int = integrate_fun(hcov_sum, low, upp)[0]
        full_int += piec_int * 2/np.pi*np.arctan(width/(2*mid))

    std_err = np.sqrt(2*np.pi*full_int / area)

    if crange2 is None or psill2 is None or model2 is None:
        return psill1 / std_err ** 2
    else:
        return (psill1 + psill2) / std_err ** 2


def integrate_fun(fun,low_limit,up_limit):

    return integrate.quad(fun,low_limit,up_limit)

def cov(h,crange,model='Sph',psill=1.,kappa=1/2,nugget=0):

    return (nugget + psill) - vgm(h,crange,model=model,psill=psill,kappa=kappa)

def vgm(h,crange,model='Sph',psill=1.,kappa=1/2,nugget=0):

    c0 = nugget #nugget
    c1 = psill #partial sill
    a1 = crange #correlation range
    s = kappa #smoothness parameter for Matern class

    if model == 'Sph':  # spherical model
        if h < a1:
            vgm = c0 + c1 * (3 / 2 * h / a1-1 / 2 * (h / a1) ** 3)
        else:
            vgm = c0 + c1
    elif model == 'Exp':  # exponential model
        vgm = c0 + c1 * (1-np.exp(-h / a1))
    elif model == 'Gau':  # gaussian model
        vgm = c0 + c1 * (1-np.exp(- (h / a1) ** 2))
    elif model == 'Exc':  # stable exponential model
        vgm = c0 + c1 * (1-np.exp(-(h/ a1)**s))

    return vgm

def std_err_finite(std, Neff, neff):
    return std * np.sqrt(1 / Neff * (Neff - neff) / Neff)

def std_err(std, Neff):
    return std * np.sqrt(1 / Neff)


def distance_latlon(tup1,tup2):

    # approximate radius of earth in km
    R = 6373000

    lat1 = m.radians(abs(tup1[1]))
    lon1 = m.radians(abs(tup1[0]))
    lat2 = m.radians(abs(tup2[1]))
    lon2 = m.radians(abs(tup2[0]))

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = m.sin(dlat / 2)**2 + m.cos(lat1) * m.cos(lat2) * m.sin(dlon / 2)**2
    c = 2 * m.atan2(m.sqrt(a), m.sqrt(1 - a))

    distance = R * c

    return distance

def kernel_sph(xi,x0,a1):
    if np.abs(xi - x0) > a1:
        return 0
    else:
        return 1 - 3 / 2 * np.abs(xi-x0) / a1 + 1 / 2 * (np.abs(xi-x0) / a1) ** 3


def part_covar_sum(argsin):
    list_tuple_errs, corr_ranges, list_area_tot, list_lat, list_lon, i_range = argsin

    n = len(list_tuple_errs)
    part_var_err = 0
    for i in i_range:
        for j in range(n):
            d = distance_latlon((list_lon[i], list_lat[i]), (list_lon[j], list_lat[j]))
            for k in range(len(corr_ranges)):
                part_var_err += kernel_sph(0, d, corr_ranges[k]) * list_tuple_errs[i][k] * list_tuple_errs[j][k] * \
                           list_area_tot[i] * list_area_tot[j]

    return part_var_err

def double_sum_covar(list_tuple_errs, corr_ranges, list_area_tot, list_lat, list_lon,nproc=1):

    n = len(list_tuple_errs)

    if nproc==1:
        print('Deriving double covariance sum with 1 core...')
        var_err = 0
        for i in range(n):
            for j in range(n):
                d = distance_latlon((list_lon[i], list_lat[i]), (list_lon[j], list_lat[j]))
                for k in range(len(corr_ranges)):
                    var_err += kernel_sph(0, d, corr_ranges[k]) * list_tuple_errs[i][k] * list_tuple_errs[j][k] * \
                               list_area_tot[i] * list_area_tot[j]
    else:
        print('Deriving double covariance sum with '+str(nproc)+' cores...')
        pack_size = int(np.ceil(n/nproc))
        argsin = [(list_tuple_errs,corr_ranges,list_area_tot,list_lon,list_lat,np.arange(i,min(i+pack_size,n))) for k, i in enumerate(np.arange(0,n,pack_size))]
        pool = mp.Pool(nproc, maxtasksperchild=1)
        outputs = pool.map(part_covar_sum, argsin, chunksize=1)
        pool.close()
        pool.join()

        var_err = np.sum(np.array(outputs))

    area_tot = 0
    for j in range(len(list_area_tot)):
        area_tot += list_area_tot[j]

    var_err /= np.nansum(area_tot) ** 2

    return np.sqrt(var_err)

def point_lonlat_trans(epsg_out,list_tup):

    trans = coord_trans(False,4326,False,epsg_out)
    list_tup_out = []
    for tup in list_tup:
        point = ogr.Geometry(ogr.wkbPoint)
        point.AddPoint(tup[0],tup[1])
        point.Transform(trans)
        coord_out = point.GetPoint()[0:2]

        list_tup_out.append(coord_out)

    return list_tup_out

def get_spatial_corr(argsin):

    coords, vals, i, cutoffs, nlags, nmax = argsin

    if len(coords)>nmax:
        subset = np.random.choice(len(coords), nmax, replace=False)
        coords = coords[subset]
        vals = vals[subset]

    print('Drawing variograms for pack '+str(i+1)+' with '+str(len(coords))+' points.')

    arr_shape = (len(cutoffs),nlags)
    exps, bins, counts = (np.zeros(arr_shape)*np.nan for i in range(3))
    for i in range(len(cutoffs)):
        try:
            #commented "ignoring maxlag" in skgstat/binning.py
            V = skg.Variogram(coordinates=coords, values=vals, n_lags=nlags, maxlag=cutoffs[i], normalize=False, model='exponential')
        except:
            return np.zeros(arr_shape)*np.nan, np.zeros(arr_shape)*np.nan, np.zeros(arr_shape)*np.nan

        count = np.zeros(nlags)
        tmp_count = np.fromiter((g.size for g in V.lag_classes()), dtype=int)
        count[0:len(tmp_count)]=tmp_count

        exps[i,:] = V.experimental
        bins[i,:] = V.bins
        counts[i,:] = count

    return exps, bins, counts

def get_tinterpcorr(df,outfile,cutoffs=[10000,100000,1000000],nlags=100,nproc=1,nmax=10000):

    #df is a subset dataframe for points of interest, standardized
    #with an attribute .reg for regions, that must be close enough for UTM zones coordinates to be relevant?

    list_reg = list(set(list(df.reg.values)))
    df_out = pd.DataFrame()

    for k in range(len(list_reg)):

        print('Working on region: '+str(list_reg[k]))

        df_reg = df[df.reg == list_reg[k]]
        #this works for ICESat campaigns, might have to put into close groups of similar dates for IceBridge
        list_dates = list(set(list(df_reg.t.values)))

        list_ns = []
        list_dt = []
        list_camp = []
        list_vals = []
        list_coords = []

        bin_dt = [0, 5, 30, 60, 90, 120, 150, 200, 260, 460, 620, 820, 980, 1180, 1500, 2000, 2500]

        for i in range(len(list_dates)):

            print('Pack of dates number ' + str(i + 1) + ' out of ' + str(len(list_dates))+':' +str(list_dates[i]))

            for j in range(len(bin_dt)-1):
                print('Day spacing number '+str(j+1)+ ' out of '+str(len(bin_dt)-1)+': '+str(bin_dt[j])+ ' to '+str(bin_dt[j+1]))

                ind = np.logical_and.reduce((df_reg.t == list_dates[i],np.abs(df_reg.dt) >= bin_dt[j],np.abs(df_reg.dt)< bin_dt[j+1]))
                df_tmp = df_reg[ind]

                print('Found ' +str(len(df_tmp))+' observations')
                vals = df_tmp.dh.values

                if len(vals)>10:

                    lat = df_tmp.lat
                    lon = df_tmp.lon
                    list_tup = list(zip(lon,lat))
                    med_lat = np.median(lat)
                    med_lon = np.median(lon)

                    print('Median latitude is:'+str(med_lat))
                    print('Median longitude is:'+str(med_lon))
                    print('Transforming coordinates...')

                    epsg, _ = latlon_to_UTM(med_lat,med_lon)
                    list_tup_out = point_lonlat_trans(int(epsg),list_tup)

                    print('Estimating spatial correlation...')

                    list_coords.append(np.array(list_tup_out))
                    list_vals.append(vals)
                    list_dt.append(bin_dt[j]+0.5*(bin_dt[j+1]-bin_dt[j]))
                    list_ns.append(len(df_tmp))
                    list_camp.append(list_dates[i])

        if len(list_coords)>0:
            if nproc == 1:
                print('Processing with 1 core...')
                list_arr_exps, list_arr_bins, list_arr_counts = ([] for i in range(3))
                for i in range(len(list_coords)):
                    exps, bins, counts = get_spatial_corr((list_coords[i],list_vals[i],i,cutoffs,nlags,nmax))

                    list_arr_exps.append(exps)
                    list_arr_bins.append(bins)
                    list_arr_counts.append(counts)
            else:
                print('Processing with '+str(nproc)+' cores...')
                arglist = [(list_coords[i],list_vals[i],i,cutoffs,nlags,nmax) for i in range(len(list_coords))]
                pool = mp.Pool(nproc,maxtasksperchild=1)
                outputs = pool.map(get_spatial_corr,arglist,chunksize=1)
                pool.close()
                pool.join()

                print('Finished processing, compiling results...')

                zipped = list(zip(*outputs))

                list_arr_exps = zipped[0]
                list_arr_bins = zipped[1]
                list_arr_counts = zipped[2]

            for l in range(len(cutoffs)):
                for c in range(len(list_camp)):
                    df_var = pd.DataFrame()
                    df_var = df_var.assign(reg=[list_reg[k]]*nlags,nb_dt=[list_dt[c]]*nlags,bins=list_arr_bins[c][l,:]
                                           ,exp=list_arr_exps[c][l,:],count=list_arr_counts[c][l,:],cutoff=cutoffs[l],t=list_camp[c])
                    df_out = df_out.append(df_var)

    df_out.to_csv(outfile)


def aggregate_tinterpcorr(infile,cutoffs=[10000,100000,1000000]):

    # infile = '/home/atom/ongoing/work_worldwide/validation/tinterp_corr.csv'
    # outfile_reg = '/home/atom/ongoing/work_worldwide/validation/agg_reg_tinterp_corr.csv'
    # outfile_all = '/home/atom/ongoing/work_worldwide/validation/agg_all_tinterp_corr.csv'

    outfile_reg = os.path.join(os.path.dirname(infile),os.path.splitext(os.path.basename(infile))[0]+'_agg_reg.csv')
    outfile_all = os.path.join(os.path.dirname(infile),os.path.splitext(os.path.basename(infile))[0]+'_agg_all.csv')

    df = pd.read_csv(infile)

    # df = df[df.reg != 19]

    bin_dt = [0,5, 30, 60, 90, 120, 150, 200, 260, 460, 620, 820, 980, 1180, 1500, 2000, 2500]
    list_nb_dt = sorted(list(set(list(df.nb_dt))))

    #first aggregate campaigns by region by cutoff by dt
    list_reg = sorted(list(set(list(df.reg))))

    df_agg_reg = pd.DataFrame()
    for reg in list_reg:

        df_reg = df[df.reg == reg]

        for cutoff in cutoffs:
            df_cutoff = df_reg[df_reg.cutoff == cutoff]

            for nb_dt in list_nb_dt:

                df_nb_dt = df_cutoff[df_cutoff.nb_dt == nb_dt]

                df_nb_dt['exp_count_prod'] = df_nb_dt.exp.values * df_nb_dt['count'].values
                df_agg = df_nb_dt.groupby('bins',as_index=False)['exp_count_prod','count'].sum()
                df_agg['exp'] = df_agg['exp_count_prod'].values / df_agg['count']
                # df_agg['bins'] = df_agg.index.values
                df_agg['nb_dt'] = nb_dt
                df_agg['cutoff'] = cutoff
                df_agg['reg'] = reg

                df_agg_reg = df_agg_reg.append(df_agg)


    df_agg_reg.to_csv(outfile_reg)

    # df_agg_reg.reset_index()

    #then, aggregate regions by cutoff by dt
    df_agg_all = pd.DataFrame()
    for cutoff in cutoffs:
        df_cutoff = df_agg_reg[df_agg_reg.cutoff == cutoff]

        for nb_dt in list_nb_dt:

            df_nb_dt = df_cutoff[df_cutoff.nb_dt == nb_dt]

            df_nb_dt['exp_count_prod'] = df_nb_dt.exp.values * df_nb_dt['count'].values
            df_agg_2 = df_nb_dt.groupby('bins',as_index=False)['exp_count_prod','count'].sum()
            df_agg_2['exp'] = df_agg_2['exp_count_prod'].values / df_agg_2['count']
            # df_agg_2['bins'] = df_agg_2.index.values
            df_agg_2['nb_dt'] = nb_dt
            df_agg_2['cutoff'] = cutoff

            df_agg_all = df_agg_all.append(df_agg_2)

    df_agg_all.to_csv(outfile_all)



