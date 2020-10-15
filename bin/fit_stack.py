from __future__ import print_function
import argparse
import numpy as np
from pyddem.fit_tools import fit_stack


def _argparser():
    parser = argparse.ArgumentParser(description="Fit time series of elevation data to a stack of DEMs.")
    # things to add: input directory, master dem and/or elevation data
    # optional: output directory, number of processors, land mask, glacier mask, number of jitter iterations?
    # tolerance for fit?
    parser.add_argument('stack', type=str, action='store', help='NetCDF file of stacked DEMs to fit.')
    parser.add_argument('-te', '--extent', type=float, action='store', nargs=4,
                        help='Extent over which to limit fit, given as [xmin xmax ymin ymax]')
    parser.add_argument('-ref_dem', type=str, action='store', default=None,
                        help='Filename for input reference DEM.')
    parser.add_argument('-ref_date', type=str, action='store', default=None,
                        help='Date of reference DEM.')
    parser.add_argument('-f', '--filt_ref', type=str, action='store', default='min_max',
                        help='Type of filtering to do. One of min_max, time, or both ')
    parser.add_argument('-filt_thresh', type=float, action='store', default=None,
                        help='Maximum dh/dt from reference DEM for time filtering.')
    parser.add_argument('-inc_mask', type=str, action='store', default=None,
                        help='Filename of optional inclusion mask (i.e., land).')
    parser.add_argument('-exc_mask', type=str, action='store', default=None,
                        help='Filename of optional exclusion mask (i.e., glaciers).')
    parser.add_argument('-n', '--nproc', type=int, action='store', default=1, help='number of processors to use [1].')
    parser.add_argument('-m', '--method', type=str, action='store', default='gpr',
                        help='Fitting method. One of Gaussian Process Regression (gpr, default),'
                             + 'Ordinary Least Squares (ols), or Weighted Least Squares (wls)')
    parser.add_argument('-opt_gpr', action='store_true',
                        help='Run learning optimization in the GPR Fitting [False]')
    parser.add_argument('-filt_ls', action='store_true',
                        help='Filter least squares with a first fit [False]')
    parser.add_argument('-ci', type=float, default=0.99,
                        help='Confidence Interval to filter least squares fit [0.99]')
    parser.add_argument('-t', '--tlim', type=float, default=None, nargs=2,
                        help='Start and end years to fit time series to (default is read from input file).')
    parser.add_argument('-ts', '--tstep', type=float, default=0.25,
                        help='Temporal step (in years) for fitted stack [0.25]')
    parser.add_argument('-o', '--outfile', type=str, default='fit.nc',
                        help='File to save results to. [fit.nc]')
    parser.add_argument('-wf', '--write_filt', action='store_true',
                        help='Write filtered stack to file [False]')
    parser.add_argument('-c', '--clobber', action='store_true', default=False,
                        help="Clobber existing outfile [False].")
    parser.add_argument('--merge_dates', action='store_true',
                        help='Merge any DEMs with same acquisition date [False]')
    parser.add_argument('-d', '--dask_parallel', action='store_true',
                        help='Run with dask parallel tools [False]')
    return parser


def main():
    np.seterr(all='ignore')
    parser = _argparser()
    args = parser.parse_args()

    if args.ref_date is not None:
        args.ref_date = np.datetim64(args.ref_date)

    fit_stack(args.stack,
              fit_extent=args.extent,
              fn_ref_dem=args.ref_dem,
              ref_dem_date=args.ref_date,
              filt_ref=args.filt_ref,
              time_filt_thresh=args.filt_thresh,
              inc_mask=args.inc_mask,
              exc_mask=args.exc_mask,
              nproc=args.nproc,
              method=args.method,
              opt_gpr=args.opt_gpr,
              filt_ls=args.filt_ls,
              conf_filt_ls=args.ci,
              tlim=args.tlim,
              tstep=args.tstep,
              outfile=args.outfile,
              write_filt=args.write_filt,
              clobber=args.clobber,
              merge_date=args.merge_dates,
              dask_parallel=args.dask_parallel)


if __name__ == "__main__":
    main()
