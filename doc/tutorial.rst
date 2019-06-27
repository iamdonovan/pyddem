Running MMASTER with bias correction
====================================

This tutorial is designed to provide the basic steps for creating DEMs from raw ASTER imagery, using MMASTER.

If you are planning to run this process on many DEMs, it might be worth having a look at scripts such as
:doc:`pymmaster/bash/process_mmaster.sh`, or :doc:`pymmaster/bash/RunMicMacAster_batch.sh`, which are designed to automatically
process many, many scenes at a time.

And of course, you are always free/encouraged to write your own scripts as needed.

Downloading data
################

ASTER data can most easily obtained from `NASA Earthdata <https://search.earthdata.nasa.gov/>`_. Be sure to
search the ASTER L1A Reconstructed Unprocessed Instrument Data, not the terrain-corrected data.

Once you have found a suitable number of scenes, be sure to order the **Geotiff**-formatted data.

After some time, you'll get an e-mail containing links to download your data. **Be sure to download both the zipped image and the metadata (.met) file**.
At this point, you're ready to run MMASTER.


Running MMASTER to process ASTER DEMs
#####################################

Pre-processing/sorting
**********************

If you have downloaded one or more **strips** (i.e., more than one ASTER scene that were acquired continuously), you
can use :doc:`pymmaster/python/scripts/sort_aster_strips` to sort these scenes into strips of typically no more than three
individual scenes.

Processing
**********

From there, you can use :doc:`pymmaster/bash/WorkFlowASTER.sh` to run MMASTER and extract a DEM from the ASTER
imagery:
::

    WorkFlowASTER.sh -s SCENENAME -z "utm zone +north/south" -a -i 2

This will run the MMASTER workflows on a folder called SCENENAME (usually of the form AST_L1A_003MMDDYYYHHMMSS),
with outputs in UTM ZONE N/S. It will use version 2 of the fitting routine (more details `here </>`_), and it will
create track angle maps to be used in the bias removal stages. Support is also included for Polar Stereographic
coordinate systems; check :doc:`pymmaster/bash/WorkFlowASTER.sh` for more details, and for other command-line options.

Depending on the number of scenes, this may take some time (on our 80-core processing server at UiO it's about
1.5 hours per 3-scene strip; around 20-30 minutes for an individual scene). As stated in :doc:`setup`, it will
likely take significantly longer on most personal laptops, and thus we don't really recommend it.

Post-processing and cleaning
****************************

MicMac will produce a number of intermediate files, which most users will not find useful. Once you have finished
processing a directory, you can run :doc:`pymmaster/bash/PostProcessMicMac.sh`, which will search each image directory
in the current folder. It creates a new directory called **PROCESSED_FINAL**, with sub-directories for each image within
the current folder. In that folder, it will create the following files:

* **AST_L1A_003MMDDYYYYHHMMSS_Z.tif** - the DEM, masked to the visible image (and to successful correlation values).
* **AST_L1A_003MMDDYYYYHHMMSS_V123.tif** - the orthorectified false-color composite of ASTER bands 1-3, typically
  at a resolution of 15 m.
* **AST_L1A_003MMDDYYYYHHMMSS_HS.tif** - the hillshade of the DEM.
* **AST_L1A_003MMDDYYYYHHMMSS_CORR.tif** - the correlation score (from 0-100) for each pixel in the DEM.
* **TrackAngleMap_3N.tif, 3B.tif** - the track pointing angle for the 3N and 3B bands for each pixel in the DEM.
* **AST_L1A_003MMDDYYYHHMMSS..zip.met** - the original metadata files downloaded for each individual scenes.

You can also remove the intermediate files from the processing directories using :doc:`pymmaster/bash/CleanMicMac.sh`,
which will create a new directory called **PROCESSED_INITIAL**, with sub-directories for each image within the current
folder. In those folders, it will save only the final steps of the MicMac automask, correlation, orthoimage, and DEM,
in addition to the zipped raw data. This is a useful way to save space, especially when dealing with hundreds or even
thousands of images.

As detailed in `Girod et al (2017) <https://www.mdpi.com/2072-4292/9/7/704/>`_, this process will
remove a significant portion of the cross-track bias in the DEM, as well as improve the matching, but
some cross-track bias will remain, as well as all of the along-track bias. In order to remove the remaining
biases, we have to use the **pymmaster** python tools.

Using pymmaster to remove remaining biases
##########################################

External sources of elevation data
**********************************

In order to correct the remaining biases in the ASTER DEMs, we first need an external elevation dataset. There are number
of potential sources, in addition to any DEMs that you may have, and we have listed some options here (in no particular order):

* **SRTM** - The Shuttle Radar Topography Mission (SRTM) covers the entire globe (between 60N and 56S) at a spatial
  resolution of approximately 30m. It was acquired using a C-band radar instrument aboard the Space Shuttle Endeavour
  in February 2000, and is one of the better globally-consistent products available. It can be obtained from many
  different sources, including `Earth Explorer <https://earthexplorer.usgs.gov/>`_. A less-complete X-band product
  is available from `DLR <https://geoservice.dlr.de/web/maps/srtm:x-sar/>`__.
* **ASTER GDEM** - The ASTER Global DEM is a 30m DEM produced from a mosaic of ASTER imagery acquired before 2011. Tiles
  can be downloaded from `NASA Earthdata <https://search.earthdata.nasa.gov/>`_.
* **TanDEM-X 90m DEM** - The TanDEM-X 90m DEM is a global DEM product from TanDEM-X imagery. It is freely available from
  DLR, and more information can be found from `DLR <https://geoservice.dlr.de/web/dataguide/tdm90/>`__.
* **ArcticDEM** - The ArcticDEM covers the Arctic (most land above 60N plus all of Alaska). It is produced from high-resolution
  optical imagery, and is available at a resolution of 2m. More information, and downloads, can be found at the
  `Polar Geopspatial Center <https://www.pgc.umn.edu/data/arcticdem/>`_.
* **REMA** (Antarctica only) - The **Reference Elevation Map of Antarctica** covers Antarctica. Like the Arctic DEM, it
  is produced from high-resolution optical imagery, and is provided at a resolution of 8m. More information, and downloads,
  can be found at the `Polar Geospatial Center <https://www.pgc.umn.edu/data/rema/>`_.
* **ICESat** - At present, ICESat is supported through `pybob <https://pybob.readthedocs.io/en/stable/>`_, though the data
  must be stored in a particular format. Files containing ICESat data for each of the RGI regions can be obtained from
  one of the authors listed below. **planning to put them on Google Drive, or somewhere similar**

When selecting a reference dataset, several considerations should be made. The highest resolution datasets are not necessarily
the best options - storage and memory can be a concern, and the ASTER scenes have a resolution of at best 15m (the default
MMASTER processing is 30m). Radar-based DEMs such as the TanDEM-X 90m DEM and SRTM can have penetration biases over snow,
ice, and vegetation, which can have an affect on the bias removal process. ICESat is globally consistent, but sparse,
with relatively poor spatial coverage at lower latitudes. The important thing is to consider your application, and be sure
to familiarize yourself with the products you are using.

Masking non-stable terrain
**************************

In order to properly correct motion-related biases in the ASTER DEMs, it is important to mask non-stable terrain
(i.e., water bodies, glaciers, large landslides). The main routine in :doc:`pymmaster/python/modules/mmaster_tools`,
**mmaster_bias_removal**, is set up to accept two types of masks:

* exclusion masks (i.e., glaciers, water bodies, landslides, areas of deforestation) - areas where large changes are
  known to have occured, and will therefore mask the true bias signal. These are most easily provided as a path to
  a shapefile outlining the areas to exclude.
* inclusion masks (i.e., land) - areas where the ground is expected to be stable, such as mask outlining land areas.
  This is most easily provided as a path to a shapefile outlining the areas to include (in other words, the opposite
  of an exclusion mask).

At present, only one mask of each type is supported. If both types are provided, the resulting mask created will be the
(**symmetrical difference?**) of the two masks.

For glaciers, a good globally-complete source of data is the `Randolph Glacier Inventory <https://www.glims.org/RGI/>`_.
In some areas where large changes have occurred in the past decade or so, it may be necessary to update the glacier mask.
It's always a good idea to compare the glacier outlines provided with satellite images acquired around the same time as
the ASTER scenes (including the **orthorectified** ASTER scenes produced by MMASTER), as well as images acquired around
the same time as the reference dataset (where applicable).

For land and/or water masks, a good source of data is ...

mmaster_bias_correction.py
**********************************

Once you have the supplementary data needed, and have extracted the DEM(s) from raw ASTER imagery, you can run
:doc:`pymmaster/python/scripts/mmaster_bias_correction`. This script is most useful for areas where the reference (master)
DEM or elevation data is contained within a single, smaller file. In this case, you can run :doc:`pymmaster/python/scripts/mmaster_bias_correction`
as follows:
::

    mmaster_bias_correction.py path/to/reference_dem.tif AST* -a path/to/exc_mask -b path/to/inc_mask

This will run on each subdirectory of the form AST\*, masking unstable terrain using both an inclusion and an exclusion mask,
as discussed above. It will run each directory in sequence, printing the log to stdout. Results will be stored in the
default folder *biasrem* within each subdirectory.

bias_correct_tiles.py
*****************************

For DEMs, or groups of DEMs, that cover a larger area, you can run the bias correction routines by using DEM tiles, rather
than a single DEM covering the whole region. Before running :doc:`pymmaster/python/scripts/bias_correct_tiles`, you
first have to produce a shapefile of the extents of the DEM tiles. You can do this by first navigating to the directory
where you have stored the DEM tiles, and running `image_footprint.py <https://pybob.readthedocs.io/en/stable/scripts/image_footprint.html>`_:
::

    image_footprint.py *.tif -o Reference_Tiles.shp

This will produce a shapefile, Reference_Tiles.shp, which contains a footprint for each .tif file in the directory,
as well as attributes with the path and filename of each DEM tile. :doc:`pymmaster/python/scripts/bias_correct_tiles`
will use this path, and the shapefile, to create a Virtual Dataset (vrt) with each of the files - thus, it's important
to note that the tiles have the same spatial reference system.

With a shapefile of reference DEM tiles, you can run :doc:`pymmaster/python/scripts/bias_correct_tiles` in the
**PROCESSED_FINAL** directory created earlier. The following example will run on all folders in **PROCESSED_FINAL**,
using 4 cores and writing results to a subdirectory of each DEM folder called biasrem:
::

    bias_correct_tiles.py path/to/Reference_Tiles.shp AST* -a path/to/exclusion_mask -b path/to/inclusion_mask -n 4 -o biasrem

When using more than one core, :doc:`pymmaster/python/scripts/bias_correct_tiles` will, like  :doc:`pymmaster/python/scripts/bias_correct_tiles`,
write a log for each directory to a log file in that directory, so as to not clutter up your screen with multiple outputs
from multiple DEMs at the same time. The end results will be the same as when running :doc:`pymmaster/python/scripts/mmaster_bias_correction` -
the only difference is the input reference DEM.

Good luck!



