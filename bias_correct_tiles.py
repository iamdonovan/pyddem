#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
import multiprocessing as mp
from glob import glob
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import geopandas as gpd
import gdal
from shapely.strtree import STRtree
from mmaster_tools import mmaster_bias_removal, get_aster_footprint


def batch_wrapper(arg_dict):
    return mmaster_bias_removal(**arg_dict)

#def main():
np.seterr(all='ignore')
parser = argparse.ArgumentParser(description="Run MMASTER post-processing bias corrections, given external elevation data.")
# things to add: input directory, master dem and/or elevation data
# optional: output directory, number of processors, land mask, glacier mask, number of jitter iterations?
# tolerance for fit?
parser.add_argument('master_tiles', type=str, help='Shapefile of master DEM footprints to be used for bias correction')
parser.add_argument('indir', action='store', nargs='+', type=str,
                    help="directory/directories where final, georeferenced images are located.")
parser.add_argument('-s', '--slavedem', type=str, default=None,
                    help='(optional) name of DEM to correct. By default, mmaster_bias_correction.py\
                    looks for MMASTER DEMs of the form AST_L1A_..._Z.tif')
parser.add_argument('-a', '--exc_mask', type=str, default=None,
                    help='exclusion mask. Areas inside of this shapefile (i.e., glaciers)\
                    will not be used for coregistration [None]')
parser.add_argument('-b', '--inc_mask', type=str, default=None,
                    help='inclusion mask. Areas outside of this mask (i.e., water)\
                         will not be used for coregistration. [None]')
parser.add_argument('-n', '--nproc', type=int, default=1, help='number of processors to use [1].')
parser.add_argument('-o', '--outdir', type=str, default='biasrem',
                    help='directory to output files to (creates if not already present). [.]')
parser.add_argument('-p', '--points', action='store_true', default=False,
                    help="process assuming that master DEM is point elevations [False].")
parser.add_argument('-l', '--log', action='store_true', default=False,
                    help="write output to a log file rather than printing to the screen [False].")
args = parser.parse_args()

# figure out if we have one image or many
print('Number of image directories given: {}'.format(len(args.indir)))


# open master_tiles, set up a search tree
master_tiles = gpd.read_file(args.master_tiles)
s = STRtree([f for f in master_tiles['geometry'].values])

master_list = []
# for each directory in indirs, get a footprint from the met files, reproject to 3413, and buffer by 1000m.
for indir in args.indir:
    dname = os.path.basename(indir.strip(os.path.sep))  # get the actual granule/folder name
    fprint = get_aster_footprint(indir, '3413', indir=indir, polyout=False)
    res = s.query(fprint.buffer(1000))
    # get only the results that intersect our buffered footprint
    intersects = [c for c in res if fprint.buffer(1000).intersection(c).area > 0]
    fnames = [master_tiles['filename'][master_tiles['geometry'] == c].values[0] for c in intersects]
    paths = [master_tiles['path'][master_tiles['geometry'] == c].values[0] for c in intersects]
    subfolders = [master_tiles['subfolder'][master_tiles['geometry'] == c].values[0] for c in intersects]
    
    tilelist = [os.path.sep.join([paths[i], subfolders[i], f]) for i, f in enumerate(fnames)]
    # create a temporary VRT from the tiles
    gdal.BuildVRT(os.path.sep.join([indir, 'tmp_{}.vrt'.format(dname)]), tilelist, resampleAlg='bilinear')
    # print(os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)]))
    master_list.append(os.path.sep.join([os.path.abspath(indir), 'tmp_{}.vrt'.format(dname)]))

# only go through the trouble of setting up multiprocessing
# if we have more than one directory to work on.
if args.nproc > 1 and len(args.indir) > 1:
    if args.nproc > mp.cpu_count():
        print("{} cores specified to use, but I could only find \
              {} cores on this machine, so I'll use those.'".format(args.nproc, mp.cpu_count()))
        args.nproc = mp.cpu_count
    pool = mp.Pool(args.nproc)
    # get a dictionary of arguments for each of the different DEMs,
    # starting with the common arguments (master dem, glacier mask, etc.)
    arg_dict = {'glacmask': args.exc_mask, 
                'landmask': args.inc_mask, 
                'pts': args.points, 
                'out_dir': args.outdir,
                'return_geoimg': False,
                'write_log': True}
    u_args = [{'work_dir': d, 'slv_dem': '{}_Z.tif'.format(d), 'mst_dem': master_list[i]} for i, d in enumerate(args.indir)]
    for d in u_args:
        d.update(arg_dict)
    
    pool.map(batch_wrapper, u_args)
    pool.close()
    #pool.join()

else:
    odir = os.getcwd()
    for i, indir in enumerate(args.indir):
        print('Running bias correction on {}'.format(indir))
        os.chdir(indir)
        #print(os.getcwd())
        if args.slavedem is None:
            flist = glob('AST*_Z.tif')
            this_slavedem = flist[0]
            mmaster_bias_removal(master_list[i],
                                 this_slavedem,
                                 glacmask=args.exc_mask,
                                 landmask=args.inc_mask,
                                 out_dir=args.outdir,
                                 pts=args.points,
                                 return_geoimg=False,
                                 write_log=args.log)
        else:
            mmaster_bias_removal(master_list[i],
                                 args.slavedem,
                                 glaciermask=args.exc_mask,
                                 landmask=args.inc_mask,
                                 out_dir=args.outdir,
                                 pts=args.points,
                                 return_geoimg=False,
                                 write_log=args.log)
        os.chdir(odir)

# clean up after ourselves, remove the vrts we created.
for v in master_list:
    os.remove(v)


#if __name__ == "__main__":
#    main()