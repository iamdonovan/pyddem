"""
example of script to wrap the fit function: GP regression with ASTER measurement error/kernel parameters
"""
from __future__ import print_function
from pyddem.fit_tools import fit_stack, get_full_dh
from glob import glob
import os
import numpy as np
import pandas as pd
import gdal
import xarray as xr
from pyddem.vector_tools import SRTMGL1_naming_to_latlon, latlon_to_UTM

method='gpr'
subspat = None
ref_dem_date=np.datetime64('2015-01-01')
gla_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/regions/rgi60_merge.shp'
inc_mask = '/data/icesat/travail_en_cours/romain/data/outlines/rgi60/buffered/rgi60_buff_10.shp'
write_filt=True
nproc=16
clobber=True
tstep=0.1
opt_gpr=False
kernel=None
filt_ref='both'
filt_ls=False
conf_filt_ls=0.99
#specify the exact temporal extent needed to be able to merge neighbouring stacks properly
tlim=[np.datetime64('2000-01-01'),np.datetime64('2019-01-01')]

# dir_stacks='/data/icesat/travail_en_cours/romain/data/stacks/06_rgi60/'
dir_stacks = '/calcul/santo/hugonnet/worldwide/18_rgi60/stacks'
ref_dir = '/calcul/santo/hugonnet/worldwide/18_rgi60/ref'
ref_gla_csv = '/data/icesat/travail_en_cours/romain/ww_tvol_study/worldwide/18_rgi60/cov/list_glacierized_tiles_06_rgi60.csv'
df = pd.read_csv(ref_gla_csv)
tilelist = df['Tile_name'].tolist()

for tile in tilelist:

    lat, lon = SRTMGL1_naming_to_latlon(tile)
    epsg, utm = latlon_to_UTM(lat, lon)

    print('Fitting tile: ' + tile + ' in UTM zone ' + utm)

    # reference DEM
    ref_utm_dir = os.path.join(ref_dir, utm)
    ref_vrt = os.path.join(ref_utm_dir, 'tmp_' + utm + '.vrt')
    ref_list = glob(os.path.join(ref_utm_dir, '**/*.tif'), recursive=True)
    if not os.path.exists(ref_vrt):
        gdal.BuildVRT(ref_vrt, ref_list, resampleAlg='bilinear')

    dir_utm_stacks=os.path.join(dir_stacks,utm)

    fn_stack = os.path.join(dir_utm_stacks,tile+'.nc')
    outfile=os.path.join(dir_utm_stacks,tile+'_final.nc')

    fit_stack(fn_stack,fit_extent=subspat,fn_ref_dem=ref_vrt,ref_dem_date=ref_dem_date,gla_mask=gla_mask,tstep=tstep,tlim=tlim,inc_mask=inc_mask,filt_ref=filt_ref,time_filt_thresh=[-30,5],write_filt=True,outfile=outfile,method=method,filt_ls=filt_ls,conf_filt_ls=conf_filt_ls,nproc=nproc,clobber=True)

    ds = xr.open_dataset(outfile)

    t0 = np.datetime64('2000-09-01')
    t1 = np.datetime64('2019-09-01')

    get_full_dh(ds, t0, t1, os.path.join(os.path.dirname(outfile), os.path.basename(outfile)))

print('Fin.')