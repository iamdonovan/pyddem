from __future__ import print_function
import argparse
import numpy as np
from pyddem.fit_tools import fit_stack

def _argparser():
    parser = argparse.ArgumentParser(description="Fit time series of elevation data using Gaussian Process.")
    # things to add: input directory, master dem and/or elevation data
    # optional: output directory, number of processors, land mask, glacier mask, number of jitter iterations?
    # tolerance for fit?
    parser.add_argument('stack', type=str, help='NetCDF file of stacked DEMs to fit.')
    parser.add_argument('-b', '--inc_mask', type=str, default=None,
                        help='inclusion mask. Areas outside of this mask (i.e., water)\
                             will be omitted from the fitting. [None]')
    parser.add_argument('-n', '--nproc', type=int, default=1, help='number of processors to use [1].')
    parser.add_argument('-t', '--time_range', type=float, default=None, nargs=2,
                        help='Start and end dates to fit time series to (default is read from input file).')
    parser.add_argument('-o', '--outfile', type=str, default='fitted_stack.nc',
                        help='File to save results to. [fitted_stack.nc]')
    parser.add_argument('-c', '--clobber', action='store_true', default=False,
                        help="Clobber existing outfile [False].")
    return parser


def main():
    np.seterr(all='ignore')
    parser = _argparser()
    args = parser.parse_args()

    fit_stack(args.stack,
              inc_mask=args.inc_mask,
              nproc=args.nproc,
              trange=args.time_range,
              outfile=args.outfile,
              clobber=args.clobber)


if __name__ == "__main__":
    main()
