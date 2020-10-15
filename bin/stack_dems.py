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
    parser.add_argument('-do_coreg', action='store_true',
                        help='Co-register DEMs to a reference DEM before adding to stack [False]')
    parser.add_argument('-r', '--ref_tiles', action='store', type=str, default=None,
                        help='Path to shapefile of reference DEM tiles to use for co-registration [None]')
    parser.add_argument('-inc_mask', action='store', type=str, default=None,
                        help='Filename of inclusion mask (i.e., land) for use in co-registration.')
    parser.add_argument('-exc_mask', action='store', type=str, default=None,
                        help='Filename of exclusion mask (i.e., glaciers) for use in co-registration.')
    parser.add_argument('-outdir', action='store', type=str, default='tmp',
                        help='Output directory for temporary files [tmp].')
    parser.add_argument('-filt_dem', action='store', type=str, default=None,
                        help='')
    parser.add_argument('-add_ref', action='store_true',
                        help='Add reference DEM as a stack variable [False]')
    parser.add_argument('-add_corr', action='store_true',
                        help='Add correlation masks as a stack variable [False]')
    parser.add_argument('-nd', '--latlontile_nodata', action='store', type=str, default=None,
                        help='Apply nodata for a lat/lon tile footprint to avoid overlapping and simplify xarray merging.')
    parser.add_argument('-filt_mm_corr', action='store_true',
                        help='Filter MMASTER DEM with correlation mask when stacking.')
    parser.add_argument('-z', '--l1a_zipped', action='store_true',
                        help='Use if DEMs are zipped.')
    parser.add_argument('-y', '--year0', action='store', type=float, default=1900.,
                        help='Year 0 to reference time variable to [1900]')
    parser.add_argument('-t', '--tmptag', action='store', type=str, default=None,
                        help='Tag to append to temporary filenames [None]')

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
                               uncert=args.uncert,
                               coreg=args.do_coreg,
                               ref_tiles=args.ref_tiles,
                               exc_mask=args.exc_mask,
                               inc_mask=args.inc_mask,
                               outdir=args.outdir,
                               filt_dem=args.filt_dem,
                               add_ref=args.add_ref,
                               add_corr=args.add_cor,
                               latlontile_nodata=args.ref_nodata,
                               filt_mm_corr=args.filt_corr,
                               l1a_zipped=args.zipped,
                               y0=args.year0,
                               tmptag=args.tmptag)
    nco.close()


if __name__ == "__main__":
    main()