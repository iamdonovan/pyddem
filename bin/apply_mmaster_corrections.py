#!/usr/bin/env python
from __future__ import print_function
import argparse
import numpy as np
import pymmaster.mmaster_tools as mt
from pybob.GeoImg import GeoImg


def read_params_file(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    return np.array([np.float64(l.strip()) for l in lines])


def _argparser():
    parser = argparse.ArgumentParser(description="Apply MMASTER post-processing bias corrections to a MMASTER DEM.")
    parser.add_argument('dem', type=str, help='Filename of DEM to load.')
    parser.add_argument('-i', '--input_dir', type=str, default='.',
                        help='Input directory to use. Default is current working directory.')
    parser.add_argument('-m', '--corr_mask', type=str, default=None,
                        help='(optional) filename of correlation mask to apply.')
    parser.add_argument('-t', '--threshold', type=float, default=60.,
                        help='(optional) correlation threshold to use. Default is 60.')
    parser.add_argument('-c', '--cross_track_params', type=str,
                        default='biasrem/params_CrossTrack_Polynomial.txt',
                        help='(Relative) path to cross-track correction parameter file. Default is\
                        [input_dir]/biasrem/params_CrossTrack_Polynomial.txt')
    parser.add_argument('-l', '--low_freq', action='store_true', default=False,
                        help='Only apply low-frequency along-track corrections. Default applies full frequency.')
    parser.add_argument('-f', '--along_track_full', type=str,
                        default='biasrem/params_AlongTrack_SumofSines.txt',
                        help='(Relative) path to full-frequency along-track correction parameter file. Default is\
                        [input_dir]/biasrem/params_AlongTrack_SumofSines.txt')
    parser.add_argument('-w', '--along_track_low', type=str,
                        default='biasrem/params_AlongTrack_SumofSines_lowfreq.txt',
                        help='(Relative) path to low-frequency along-track correction parameter file. Default is\
                        [input_dir]/biasrem/params_AlongTrack_SumofSines_lowfreq.txt')
    parser.add_argument('-o', '--outfilename', type=str, default=None,
                        help='Output filename for corrected DEM. Default is inputdem_XAJ.tif')
    return parser


def main():
    np.seterr(all='ignore')
    parser = _argparser()
    args = parser.parse_args()

    if args.outfilename is None:
        args.outfilename = args.dem.rsplit('.tif', 1)[0] + '_XAJ.tif'
    # read in DEM to apply corrections to
    dem = GeoImg(args.dem)
    
    # apply optional correlation mask
    if args.corr_mask is not None:
        corr = GeoImg(args.corr_mask)
        dem.img[corr.img < args.threshold] = np.nan

    ang_mapN = GeoImg('TrackAngleMap_3N.tif')
    ang_mapB = GeoImg('TrackAngleMap_3B.tif')
    ang_mapNB = ang_mapN.copy(new_raster=np.array(np.divide(ang_mapN.img + ang_mapB.img, 2)))

    # first apply the cross-track correction
    myang = np.deg2rad(ang_mapNB.img)
    xxr, _ = mt.get_xy_rot(dem, myang)
    pcoef = read_params_file(args.cross_track_params)
    cross_correction = mt.fitfun_polynomial(xxr, pcoef)
    dem.img = dem.img + cross_correction

    # next, apply *either* the full soluton, or just the low-frequency solution
    xxn_mat, xxb_mat = mt.get_atrack_coord(dem, ang_mapN, ang_mapB)
    if not args.low_freq:
        scoef = read_params_file(args.along_track_full)
    else:
        scoef = read_params_file(args.along_track_low)
    sinmod = mt.fitfun_sumofsin_2angle(xxn_mat, xxb_mat, scoef)
    along_correction = np.reshape(sinmod, dem.img.shape)
    dem.img = dem.img + along_correction
    dem.write(args.outfilename, out_folder=args.input_dir)


if __name__ == "__main__":
    main()