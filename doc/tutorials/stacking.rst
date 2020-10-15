creating a stack of dems with pyddem
====================================

The basic procedure for stacking DEMs involves using either :doc:`../pyddem/modules/stack_tools#pyddem.stack_tools.create_mmaster_stack`
or :doc:`../pyddem/scripts/stack_dems` from the command line.

The first step is to have a number of DEMs available to stack. For this example, we'll assume that these are in the same
folder as we are trying to create the stack, and that they all have a filename of the form **AST*.tif**. Note that any
file format that can be read by `GDAL <https://gdal.org>`__ will work here, but the date of acquisition should be parse-able
by **pybob.GeoImg**.

Using a reference DEM
#####################
If the DEMs are not already co-registered to a reference DEM, you can do that within the stacking process.

To do this, you need to have a shapefile (or any file format that can be opened with `geopandas <https://geopandas.org>`__)
with the footprint(st) of the reference DEM. At the moment, this is set up assuming that the tiles are stored in various
'subfolder's of a single 'path'. So, each feature of the features in the shapefile should have attributes that look
something like this:

+-----+--------------------------------+---------------+------------------------------+
| ID  | filename                       | subfolder     | path                         |
+=====+================================+===============+==============================+
| 0   | 59_25_1_2_30m_v3.0_reg_dem.tif | NorthAsia     | ~/data/ArcticDEM/v3.0/Mosaic |
+-----+--------------------------------+---------------+------------------------------+

With this, create_mmaster_stack() will create a VRT from all tiles that intersect the boundary of the stack, and use this
to co-register each individual DEM in turn. You can also pass filenames for an exclusion mask (i.e., glaciers) to identify
terrain that should not be included in the co-registration, as well as an inclusion mask (i.e., land).

Running from a script
#####################
The following example demonstrates how you can run :doc:`../pyddem/modules/stack_tools#pyddem.create_mmaster_stack`
from your own script. create_mmaster_stack() will load each of the DEMs in turn and insert it into the stack.
::
    from glob import glob
    from pyddem.stack_tools import create_mmaster_stack

    filelist_dems = glob('AST*.tif')
    fn_out = 'my_stack.nc'
    fn_glacmask = '~/data/RGI/v6.0/10_rgi60_NorthAsia/10_rgi60_NorthAsia.shp'
    fn_reftiles = '~/data/ArcticDEM/v3.0/Mosaic/ArcticDEM_Tiles.shp'

    my_stack = create_mmaster_stack(filelist_dems,
                                    res=100,
                                    outfile=fn_out,
                                    exc_mask=fn_glacmask,
                                    mst_tiles=fn_reftiles,
                                    clobber=True,
                                    add_ref=True,
                                    coreg=True)

This will create an output file called **my_stack.nc**. Note that by not specifying an extent, we automatically
take the extent of the first DEM (chronologically). In the stack file, each of the **AST*.tif** DEMs are re-sampled to 100 m
(**res=100**) and added to the stack in chronological order. It will co-register each of the DEMs to the reference
DEM (**coreg=True**), overwrite any existing file (**clobber=True**), and add the reference DEM to the stack
(**add_ref**), with any terrain falling within the RGI v6.0 glacier outlines (**exc_mask=fn_glacmask**)
ignored for co-registration.

Running from the command line
#############################
We can do the same thing as above using the command-line script provided in **bin/stack_dems.py**:
::
    stack_dems.py AST*.tif -res 100 -o my_stack.nc -exc_mask ~/data/RGI/v6.0/10_rgi60_NorthAsia/10_rgi60_NorthAsia.shp
            -ref_tiles '~/data/ArcticDEM/v3.0/Mosaic/ArcticDEM_Tiles.shp' -c -do_coreg -add_ref

Once you have a stack of DEMs, you can run various :doc:`fitting` routine on the stack to fit a time series of elevation.