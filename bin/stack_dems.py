#! /usr/bin/env python
from __future__ import print_function
import argparse
from pyddem.stack_tools import create_mmaster_stack


def _argparser():
    # have to add support for co-registration - check docs for stack_toosl.create_mmaster_stack()
    parser = argparse.ArgumentParser(description="Create a NetCDF stack of DEMs.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filelist', action='store', type=str, nargs='+',
                        help='List of DEM files to read and stack.')
    parser.add_argument('-extent', action='store', metavar=('xmin', 'xmax', 'ymin', 'ymax'),
                        type=float, nargs=4, help='Extent of output DEMs to write.')
    parser.add_argument('-res', action='store', type=float, default=None,
                        help='pixel resolution (in meters)')
    parser.add_argument('-epsg', action='store', type=int, default=None,
                        help='Target EPSG code. Default is taken from first (chronological) DEM.')
    parser.add_argument('-o', '--outfile', action='store', type=str, default='mmaster_stack.nc',
                        help='Output NetCDF file to create [mmaster_stack.nc]')
    parser.add_argument('-c', '--clobber', action='store_true',
                        help='Overwrite any existing file [False]')
    parser.add_argument('-u', '--uncert', action='store_true',
                        help='Read stable terrain statistics for each DEM from [filename].txt')
    return parser


def main():
    parser = _argparser()
    args = parser.parse_args()

    nco = create_mmaster_stack(args.filelist,
                               extent=args.extent,
                               res=args.res,
                               epsg=args.epsg,
                               outfile=args.outfile,
                               clobber=args.clobber,
                               uncert=args.uncert)
    nco.close()


if __name__ == "__main__":
    main()