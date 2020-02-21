"""
pyddem.stack_tools provides tools to create stacks of DEM data, usually MMASTER DEMs.
"""
from __future__ import print_function
import os, sys
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6
import time
# import errno
# import geopandas as gpd
import numpy as np
import datetime as dt
import gdal
# import ogr
# import gdalconst
import netCDF4
import geopandas as gpd
import xarray as xr
from shapely.geometry.polygon import Polygon
from shapely.ops import cascaded_union
from shapely.strtree import STRtree
from osgeo import osr
from skimage.morphology import disk
from scipy.ndimage.morphology import binary_opening
from pybob.GeoImg import GeoImg
import pymmaster.mmaster_tools as mt
import pymmaster.other_tools as ot
from pybob.coreg_tools import dem_coregistration
from pybob.bob_tools import mkdir_p


def read_stats(fdir):

    #search for the stats file in different directory levels
    list_poss=[os.path.join(fdir,'re-coreg','stats.txt'),os.path.join(fdir,'coreg','stats.txt'),os.path.join(fdir,'stats.txt')]
    fname = None
    for poss_fname in list_poss:
        if os.path.exists(poss_fname):
            fname = poss_fname
            break

    if fname is None:
        print('Could not find a stats.txt file in re-coreg, coreg or root of: '+fdir)
        sys.exit()

    with open(fname, 'r') as f:
        lines = f.readlines()
    stats = lines[0].strip('[ ]\n').split(', ')
    after = [float(s) for s in lines[-1].strip('[ ]\n').split(', ')]

    return dict(zip(stats, after))

def corr_filter_aster(fn_dem,fn_corr,threshold=80):

    dem = GeoImg(fn_dem)
    corr = GeoImg(fn_corr)
    corr.img[corr.img < threshold] = 0

    rem_open = binary_opening(corr.img, structure=disk(5))
    dem.img[~rem_open] = np.nan

    return dem

def parse_date(fname, datestr=None, datefmt=None):
    bname = os.path.splitext(os.path.basename(fname))[0]
    splitname = bname.split('_')
    if datestr is None and datefmt is None:
        if splitname[0] == 'AST':
            datestr = splitname[2][3:]
            datefmt = '%m%d%Y%H%M%S'
        elif splitname[0] in ['SETSM', 'SDMI', 'SPOT5', 'Map', 'IDEM', 'S0', 'USGS', 'CDEM', 'TDM1']:
            datestr = splitname[2]
            datefmt = '%Y%m%d'
        elif splitname[0] in ['aerodem']:
            datestr = splitname[1]
            datefmt = '%Y%m%d'
        else:
            print("I don't recognize how to parse date information from {}.".format(fname))
            return None
    return dt.datetime.strptime(datestr, datefmt)

#faster OGR solution for rasters + read l1a metadata if files are zipped
def get_footprints_inters_ext(filelist, extent_base, epsg_base, use_l1a_met=False):

    list_poly = []
    for f in filelist:
        if use_l1a_met:
            poly=ot.l1astrip_polygon(os.path.dirname(f))
            trans=ot.coord_trans(False,4326,False,epsg_base)
            poly.Transform(trans)
        else:
            ext, proj = ot.extent_rast(f)
            poly = ot.poly_from_extent(ext)
            trans= ot.coord_trans(True,proj,False,epsg_base)
            poly.Transform(trans)
        list_poly.append(poly)

    poly_ext = ot.poly_from_extent(extent_base)

    filelist_out = []
    for poly in list_poly:
        inters = poly.Intersection(poly_ext)
        if not inters.IsEmpty():
            filelist_out.append(filelist[list_poly.index(poly)])

    return filelist_out


def get_footprints(filelist, proj4=None):
    """
    Get a list of footprints, given a filelist of DEMs.

    :param filelist: List of DEMs to create footprints for.
    :param proj4: proj4 representation of output CRS. If None, the CRS is chosen from the first DEM loaded. Can also supply
        an EPSG code as an integer.
    :type filelist: array-like
    :type proj4: str, int

    :returns fprints, this_crs: A list of footprints and a proj4 string (or dict) representing the output CRS.
    """
    fprints = []
    if proj4 is not None:
        if type(proj4) is int:
            this_proj4 = {'init': 'epsg:{}'.format(proj4)}
        else:
            this_proj4 = proj4
    else:
        tmp = GeoImg(filelist[0])
        this_proj4 = tmp.proj4

    for f in filelist:
        tmp = GeoImg(f)
        fp = Polygon(tmp.find_corners(mode='xy'))
        fprints.append(mt.reproject_geometry(fp, tmp.proj4, this_proj4))

    return fprints, this_proj4


def get_common_bbox(filelist, epsg=None):
    fprints, _ = get_footprints(filelist, epsg)

    common_fp = cascaded_union(fprints)
    bbox = common_fp.envelope

    x, y = bbox.boundary.coords.xy
    return min(x), max(x), min(y), max(y)


def create_crs_variable(epsg, nco):
    """
    Given an EPSG code, create a CRS variable for a NetCDF file.

    :param epsg: EPSG code for chosen CRS.
    :param nco: NetCDF file to create CRS variable for.
    :type epsg: int
    :type nco: netCDF4.Dataset
    :returns crso: NetCDF variable representing the CRS.
    """
    sref = osr.SpatialReference()
    sref.ImportFromEPSG(epsg)

    sref_wkt = sref.ExportToWkt()

    crso = nco.createVariable('crs', 'S1')
    crso.long_name = sref_wkt.split(',')[0].split('[')[-1].replace('"', '').replace(' / ', ' ')

    if 'PROJCS' in sref_wkt:
        split1 = sref_wkt.split(',PROJECTION')
        split2 = split1[1].split('],')
        split3 = split1[0].split(',GEOGCS[')[1]

        crso.grid_mapping_name = split2[0].strip('["').lower()

        params = [s.split('[')[-1] for s in split2 if 'PARAMETER' in s]
        for p in params:
            if p.split(',')[0].strip('"') == 'scale_factor':
                exec('crso.{} = {}'.format('proj_scale_factor', float(p.split(',')[1])))
            else:
                exec('crso.{} = {}'.format(p.split(',')[0].strip('"'), float(p.split(',')[1])))
        ustr = [s.split('[')[1].split(',')[0].strip('"') for s in split2 if 'UNIT' in s][0]
        crso.units = ustr
        crso.spheroid = split3.split(',SPHEROID[')[0].split(',')[0].strip('"').replace(' ', '')
        crso.semi_major_axis = float(split3.split(',SPHEROID')[1].split(',')[1])
        crso.inverse_flattening = float(split3.split(',SPHEROID')[1].split(',')[2])
        crso.datum = split3.split(',SPHEROID[')[0].split(',DATUM[')[-1].strip('"')
        crso.longitude_of_prime_meridian = int(split3.split(',PRIMEM')[-1].split(',')[1])
        if crso.grid_mapping_name == 'polar_stereographic':
            crso.scale_factor_at_projection_origin = crso.proj_scale_factor
            crso.standard_parallel = crso.latitude_of_origin
        # crso.spheroid = 'WGS84'
        # crso.datum = 'WGS84'
        crso.spatial_ref = sref_wkt
    else:  # have to figure out what to do with a non-projected system...
        pass

    return crso


def create_nc(img, outfile='mmaster_stack.nc', clobber=False, t0=None):
    """
    Create a NetCDF dataset with x, y, and time variables.

    :param img: Input GeoImg to base shape of x, y variables on.
    :param outfile: Filename for output NetCDF file.
    :param clobber: clobber existing dataset when creating NetCDF file.
    :param t0: Initial time for creation of time variable. Default is 01 Jan 1900.
    :type img: pybob.GeoImg
    :type outfile: str
    :type clobber: bool
    :type t0: np.datetime64
    :returns nco, to, xo, yo: output NetCDF dataset, time, x, and y variables.
    """
    nrows, ncols = img.shape

    # nc file creation fails if we don't create manually the parent directory
    outdir = os.path.dirname(outfile)
    if outdir == '':
        outdir = '.'
    mkdir_p(outdir)

    nco = netCDF4.Dataset(outfile, 'w', clobber=clobber)
    nco.createDimension('x', ncols)
    nco.createDimension('y', nrows)
    nco.createDimension('time', None)
    nco.Conventions = 'CF-1.7'
    nco.description = "Stack of co-registered DEMs produced using MMASTER (+ other sources). \n" + \
                      "MMASTER scripts and documentation: https://github.com/luc-girod/MMASTER-workflows \n" + \
                      "pybob source and documentation: https://github.com/iamdonovan/pybob"
    nco.history = "Created " + time.ctime(time.time())
    nco.source = "Robert McNabb (robertmcnabb@gmail.com)"

    to = nco.createVariable('time', 'f4', ('time'))
    if t0 is None:
        to.units = 'days since 1900-01-01'
    else:
        to.units = 'days since {}'.format(np.datetime_as_string(t0))
    to.calendar = 'standard'
    to.standard_name = 'date'

    xo = nco.createVariable('x', 'f4', ('x')) #, chunksizes=[10])
    xo.units = 'm'
    xo.standard_name = 'projection_x_coordinate'
    xo.axis = 'X'

    yo = nco.createVariable('y', 'f4', ('y')) # chunksizes=[10])
    yo.units = 'm'
    yo.standard_name = 'projection_y_coordinate'
    yo.axis = 'Y'

    return nco, to, xo, yo


def get_tiles(bounds, master_tiles, s, name):
    res = s.query(bounds)
    # get only the results that intersect our buffered footprint more than 10% of its area
    intersects = [c for c in res if bounds.intersection(c).area / c.area > 0.1]
    fnames = [master_tiles['filename'][master_tiles['geometry'] == c].values[0] for c in intersects]
    paths = [master_tiles['path'][master_tiles['geometry'] == c].values[0] for c in intersects]
    subfolders = [master_tiles['subfolder'][master_tiles['geometry'] == c].values[0] for c in intersects]

    tilelist = [os.path.sep.join([paths[i], subfolders[i], f]) for i, f in enumerate(fnames)]
    # create a temporary VRT from the tiles
    gdal.BuildVRT('{}_mst.vrt'.format(name), tilelist, resampleAlg='bilinear')
    # print(os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)]))
    return '{}_mst.vrt'.format(name)

def extent_stack(ds):

    xmin = np.round(ds.x.min().values).astype(float)
    xmax = np.round(ds.x.max().values).astype(float)
    ymin = np.round(ds.y.min().values).astype(float)
    ymax = np.round(ds.y.max().values).astype(float)

    extent = [xmin, ymin, xmax, ymax]
    proj = ds.crs.spatial_ref

    return extent, proj

def open_datasets(list_fn_stack):

    list_ds=[]
    for fn_stack in list_fn_stack:
        ds = xr.open_dataset(fn_stack)
        list_ds.append(ds)

    return list_ds

def combine_stacks(list_ds):

    #get list of crs, and choose crs the most frequent as the reference for combining all datasets
    list_crs = []
    for ds in list_ds:
        list_crs.append(ds.crs.spatial_ref)
    if len(set(list_crs)) > 1:
        crs_ref = max(set(list_crs),key=list_crs.count)
    else:
        crs_ref = list_crs[0]

    #reproject stacks to crs_ref if needed
    list_ds_commonproj = []
    for i in range(len(list_ds)):
        crs = list_crs[i]
        ds = list_ds[i]
        if not crs == crs_ref:
            ds_proj = reproj_stack(ds,crs_ref)
            list_ds_commonproj.append(ds_proj)
        else:
            list_ds_commonproj.append(ds)

    #merge all stacks
    ds_combined = merge_stacks(list_ds_commonproj)

    return ds_combined

def merge_stacks(list_ds):

    #works as we intend only if coordinates are aligned (see nice coords in other_tools) AND if no values other than NaN intersect (hence the latlontile_nodata mask in stack_tools)
    ds = xr.merge(list_ds)

    #a lot faster if tiles are not-overlapping tiles in a given projection (works as intended if xarray >= 0.12.3) ; this is impossible for lat/lon tiles in UTM
    #ds = xr.combine_by_coords(list_ds)

    return ds

def reproj_stack(ds,proj_out,niceextent=True,latlontile_nodata=True):

    pass

def make_geoimg(ds, band=0):
    """
    Create a GeoImg representation of a given band from an xarray dataset.

    :param ds: xarray dataset to read shape, extent, CRS values from.
    :param band: band number of xarray dataset to use

    :type ds: xarray.Dataset
    :type band: int
    :returns geoimg: GeoImg representation of the given band.
    """
    npix_y, npix_x = ds['z'][band].shape
    dx = np.round((ds.x.max().values - ds.x.min().values) / float(npix_x))
    dy = np.round((ds.y.min().values - ds.y.max().values) / float(npix_y))

    newgt = (ds.x.min().values - 0, dx, 0, ds.y.max().values - 0, 0, dy)

    drv = gdal.GetDriverByName('MEM')
    dst = drv.Create('', npix_x, npix_y, 1, gdal.GDT_Float32)

    sp = dst.SetProjection(ds.crs.spatial_ref)
    sg = dst.SetGeoTransform(newgt)

    wa = dst.GetRasterBand(1).WriteArray(ds['z'][band].values)
    md = dst.SetMetadata({'Area_or_point': 'Point'})
    del wa, sg, sp, md

    return GeoImg(dst)

def create_mmaster_stack(filelist, extent=None, res=None, epsg=None, outfile='mmaster_stack.nc',
                         clobber=False, uncert=False, coreg=False, mst_tiles=None,
                         exc_mask=None, inc_mask=None, outdir='tmp', filt_dem=None, add_ref=False,
                         latlontile_nodata=None, filt_mm_corr=False, l1a_zipped=False, y0=1900):
    """
    Given a list of DEM files, create a stacked NetCDF file.

    :param filelist: List of DEM filenames to stack.
    :param extent: Spatial extent of DEMs to limit stack to [xmin, xmax, ymin, ymax].
    :param res: Output spatial resolution of DEMs.
    :param epsg: EPSG code of output CRS.
    :param outfile: Filename for output NetCDF file.
    :param clobber: clobber existing dataset when creating NetCDF file.
    :param uncert: Include uncertainty variable in the output NetCDF.
    :param coreg: Co-register DEMs to an input DEM (given by a shapefile of tiles).
    :param mst_tiles: Filename of input master DEM tiles.
    :param exc_mask: Filename of exclusion mask (i.e., glaciers) to use in co-registration
    :param inc_mask: Filename of inclusion mask (i.e., land) to use in co-registration.
    :param outdir: Output directory for temporary files.
    :param filt_dem: Filename of DEM to filter elevation differences to.
    :param add_ref: Add reference DEM as a stack variable
    :param latlontile_nodata: Apply nodata for a lat/lon tile footprint to avoid overlapping and simplify xarray merging
    :param filt_mm_corr: Filter MMASTER DEM with correlation mask out of mmaster_tools when stacking (disk space)

    :type filelist: array-like
    :type extent: array-like
    :type res: float
    :type epsg: int
    :type outfile: str
    :type clobber: bool
    :type uncert: bool
    :type coreg: bool
    :type mst_tiles: str
    :type exc_mask: str
    :type inc_mask: str
    :type outdir: str
    :type filt_dem: str
    :type add_ref: bool
    :type latlontile_nodata: str
    :type filt_mm_corr: bool

    :returns nco: NetCDF Dataset of stacked DEMs.
    """
    if extent is not None:
        if type(extent) in [list, tuple]:
            xmin, xmax, ymin, ymax = extent
        elif type(extent) is Polygon:
            x, y = extent.boundary.coords.xy
            xmin, xmax = min(x), max(x)
            ymin, ymax = min(y), max(y)
        else:
            raise ValueError('extent should be a list, tuple, or shapely.Polygon')
    else:
        xmin, xmax, ymin, ymax = get_common_bbox(filelist, epsg)

    print('Searching for intersecting DEMs among the list of '+str(len(filelist))+'...')
    # check if each footprint falls within our given extent, and if not - remove from the list.
    if l1a_zipped:
        #if l1a are zipped, too long to extract archives and read extent from rasters ; so read metadata instead
        l1a_filelist = [fn for fn in filelist if os.path.basename(fn)[0:3]=='AST']
        rest_filelist = [fn for fn in filelist if fn not in l1a_filelist]
        l1a_inters = get_footprints_inters_ext(l1a_filelist,[xmin,ymin,xmax,ymax],epsg,use_l1a_met=True)
        rest_inters = get_footprints_inters_ext(rest_filelist,[xmin,ymin,xmax,ymax],epsg)
        filelist = l1a_inters + rest_inters

    else:
        filelist = get_footprints_inters_ext(filelist,[xmin,ymin,xmax,ymax],epsg)
    print('Found '+str(len(filelist))+'.')

    datelist = np.array([parse_date(f) for f in filelist])
    sorted_inds = np.argsort(datelist)

    if l1a_zipped:
        tmp_zip = filelist[sorted_inds[0]]
        z_name = '_'.join(os.path.basename(tmp_zip).split('_')[0:3]) + '_Z_adj_XAJ_final.tif'
        fn_tmp = os.path.join(os.path.dirname(tmp_zip),'tmp_out.tif')
        mt.extract_file_from_zip(tmp_zip,z_name,fn_tmp)
        tmp_img = GeoImg(fn_tmp)
    else:
        tmp_img = GeoImg(filelist[sorted_inds[0]])

    if res is None:
        res = np.round(tmp_img.dx)  # make sure that we have a nice resolution for gdal

    if epsg is None:
        epsg = tmp_img.epsg

    # now, reproject the first image to the extent, resolution, and coordinate system needed.
    dest = gdal.Warp('', tmp_img.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg),
                     xRes=res, yRes=res, outputBounds=(xmin, ymin, xmax, ymax),
                     resampleAlg=gdal.GRA_Bilinear)

    if l1a_zipped:
        os.remove(fn_tmp)

    first_img = GeoImg(dest)

    first_img.filename = filelist[sorted_inds[0]]

    # NetCDF assumes that coordinates are the cell center
    if first_img.is_area():
        first_img.to_point()
    # first_img.info()

    nco, to, xo, yo = create_nc(first_img.img, outfile=outfile, clobber=clobber, t0=np.datetime64('{}-01-01'.format(y0)))
    create_crs_variable(first_img.epsg, nco)
    # crso.GeoTransform = ' '.join([str(i) for i in first_img.gd.GetGeoTransform()])

    # maxchar = max([len(f.rsplit('.tif', 1)[0]) for f in args.filelist])
    go = nco.createVariable('dem_names', str, ('time',))
    go.long_name = 'Source DEM Filename'

    zo = nco.createVariable('z', 'f4', ('time', 'y', 'x'), fill_value=-9999)
    zo.units = 'meters'
    zo.long_name = 'Height above WGS84 ellipsoid'
    zo.grid_mapping = 'crs'
    zo.coordinates = 'x y'
    zo.set_auto_mask(True)

    if mst_tiles is not None:
        if mst_tiles.endswith('.shp'):
            master_tiles = gpd.read_file(mst_tiles)
            s = STRtree([f for f in master_tiles['geometry'].values])
            bounds = Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            mst_vrt = get_tiles(bounds, master_tiles, s, outdir)
        elif mst_tiles.endswith('.vrt') or mst_tiles.endswith('.tif'):
            mst_vrt = mst_tiles
        mst = GeoImg(mst_vrt)

    if filt_dem is not None:
        filt_dem_img = GeoImg(filt_dem)
        filt_dem = filt_dem_img.reproject(first_img)

    # make sure we have no overlapping pixels between tile: makes it SO MUCH easier to merge with xarray + speed up process
    if latlontile_nodata is not None and epsg is not None:
        mask = ot.latlontile_nodatamask(first_img, latlontile_nodata, ot.utm_from_epsg(epsg))

    if uncert:
        uo = nco.createVariable('uncert', 'f4', ('time',))
        uo.long_name = 'RMSE of stable terrain differences.'
        uo.units = 'meters'

    if add_ref and mst_tiles is not None:
        ro = nco.createVariable('ref_z','f4',('y','x'), fill_value=-9999)
        ro.units = 'meters'
        ro.long_name = 'Height above WGS84 ellipsoid'
        ro.grid_mapping = 'crs'
        ro.coordinates = 'x y'
        ro.set_auto_mask(True)
        mst_img = mst.reproject(first_img).img
        if latlontile_nodata is not None and epsg is not None:
            mst_img[~mask] = np.nan
            ro[: , :] = mst_img

    x, y = first_img.xy(grid=False)
    xo[:] = x
    yo[:] = y

    outind = 0
    for ind in sorted_inds[0:]:
        print(filelist[ind])
        #get instrument
        bname = os.path.splitext(os.path.basename(filelist[ind]))[0]
        splitname = bname.split('_')
        instru = splitname[0]
        #special case for MMASTER outputs (for disk usage)
        if instru == 'AST':
            fn_z = '_'.join(splitname[0:3]) + '_Z_adj_XAJ_final.tif'
            fn_z_tmp = os.path.join(os.path.dirname(filelist[ind]), fn_z)
            fn_corr = '_'.join(splitname[0:3]) + '_CORR_adj_final.tif'
            fn_corr_tmp = os.path.join(os.path.dirname(filelist[ind]), fn_corr)
            list_fn_rm = [fn_z_tmp, fn_corr_tmp]
            #unzip if needed
            if l1a_zipped:
                mt.extract_file_from_zip(filelist[ind],fn_z,fn_z_tmp)
                if filt_mm_corr:
                    mt.extract_file_from_zip(filelist[ind],fn_corr,fn_corr_tmp)
            #open dem, filter with correlation mask if it comes out of MMASTER
            if filt_mm_corr:
                img = corr_filter_aster(fn_z_tmp,fn_corr_tmp,70)
            else:
                img = GeoImg(fn_z_tmp)
        else:
            img = GeoImg(filelist[ind])

        if img.is_area():  # netCDF assumes coordinates are the cell center
            img.to_point()

        if coreg:
            try:
                NDV = img.NDV
                coreg_outdir = os.path.join(outdir,os.path.basename(filelist[ind]).rsplit('.tif', 1)[0])
                _, img, _ , stats_final = dem_coregistration(mst, img, glaciermask=exc_mask, landmask=inc_mask, outdir=coreg_outdir, inmem=True)
                dest = gdal.Warp('', img.gd, format='MEM', dstSRS='EPSG:{}'.format(epsg),
                                 xRes=res, yRes=res, outputBounds=(xmin, ymin, xmax, ymax),
                                 resampleAlg=gdal.GRA_Bilinear, srcNodata=NDV, dstNodata=-9999)
                img = GeoImg(dest)
                if filt_dem is not None:
                    valid = np.logical_and(img.img - filt_dem.img > -400,
                                           img.img - filt_dem.img < 1000)
                    img.img[~valid] = np.nan
                if latlontile_nodata is not None and epsg is not None:
                    img.img[~mask] = np.nan
                nvalid = np.count_nonzero(~np.isnan(img.img))
                if nvalid == 0:
                    print('No valid pixel in the stack extent: skipping...')
                    if l1a_zipped and (instru == 'AST'):
                        for fn_rm in list_fn_rm:
                            if os.path.exists(fn_rm):
                                os.remove(fn_rm)
                    continue
                zo[outind, :, :] = img.img
                if uncert:
                    uo[outind] = stats_final[3]
                print('Adding DEM that has '+str(nvalid)+' valid pixels in this extent, with a global RMSE of '+str(stats_final[3]))
            except:
                print('Coregistration failed: skipping...')
                if l1a_zipped and (instru == 'AST'):
                    for fn_rm in list_fn_rm:
                        if os.path.exists(fn_rm):
                            os.remove(fn_rm)
                continue

        else:
            img = img.reproject(first_img)
            if filt_dem is not None:
                valid = np.logical_and(img.img - filt_dem.img > -400,
                                       img.img - filt_dem.img < 1000)
                img.img[~valid] = np.nan
            if latlontile_nodata is not None and epsg is not None:
                img.img[~mask] = np.nan
            nvalid = np.count_nonzero(~np.isnan(img.img))
            if nvalid == 0:
                print('No valid pixel in the stack extent: skipping...')
                if l1a_zipped and (instru == 'AST'):
                    for fn_rm in list_fn_rm:
                        if os.path.exists(fn_rm):
                            os.remove(fn_rm)
                continue
            zo[outind, :, :] = img.img
            if uncert:
                stats = read_stats(os.path.dirname(filelist[sorted_inds[ind]]))
                uo[outind] = stats['RMSE']
        to[outind] = datelist[ind].toordinal() - dt.date(1900, 1, 1).toordinal()
        go[outind] = os.path.basename(filelist[ind]).rsplit('.tif', 1)[0]
        zo[outind, :, :] = img.img
        outind += 1

        if l1a_zipped and (instru=='AST'):
            for fn_rm in list_fn_rm:
                if os.path.exists(fn_rm):
                    os.remove(fn_rm)

    return nco
