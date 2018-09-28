#!/usr/bin/env python
from __future__ import print_function, division
import os
import argparse
from glob import glob
import numpy as np
from mmaster_tools import mmaster_bias_removal


def batch_process(args):
    pass


def main():
    np.seterr(all='ignore')
    parser = argparse.ArgumentParser(description="Run MMASTER post-processing bias corrections, given external elevation data.")
    # things to add: input directory, master dem and/or elevation data
    # optional: output directory, number of processors, land mask, glacier mask, number of jitter iterations?
    # tolerance for fit?
    parser.add_argument('masterdem', type=str, help='path to master DEM/elevations to be used for bias correction')
    parser.add_argument('indir', action='store', nargs='+', type=str,
                        help="directory/directories where final, georeferenced images are located.")
    parser.add_argument('-s', '--slavedem', type=str, default=None,
                        help='(optional) name of DEM to correct. By default, mmaster_bias_correction.py\
                        looks for MMASTER DEMs of the form AST_...._Z.tif')
    parser.add_argument('-a', '--exc_mask', type=str, default=None,
                        help='exclusion mask. Areas inside of this shapefile (i.e., glaciers)\
                        will not be used for coregistration [None]')
    parser.add_argument('-b', '--inc_mask', type=str, default=None,
                        help='inclusion mask. Areas outside of this mask (i.e., water)\
                             will not be used for coregistration. [None]')
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of processors to use [1].')
    parser.add_argument('-o', '--outdir', type=str, default='.',
                        help='directory to output files to (creates if not already present). [.]')
    parser.add_argument('-p', '--points', action='store_true', default=False,
                        help="process assuming that master DEM is point elevations [False].")
    args = parser.parse_args()
    
    # figure out if we have one image or many
    print('Number of image directories given: {}'.format(len(args.indir)))
    if len(args.indir) == 1:
        args.indir = args.indir[0]
        print('Running bias correction on {}'.format(args.indir))
        os.chdir(args.indir)
        if args.slavedem is None:
            flist = glob('AST*_Z.tif')
            args.slavedem = flist[0]

        mst_coreg, slv_coreg_xcorr_acorr_jcorr = mmaster_bias_removal(args.masterdem,
                                                                      args.slavedem,
                                                                      glacmask=args.exc_mask,
                                                                      landmask=args.inc_mask,
                                                                      out_dir=args.outdir,
                                                                      pts=args.points)
    else:
        print('not implemented yet!')
        

if __name__ == "__main__":
    main()