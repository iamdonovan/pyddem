Running MMASTER with bias correction
====================================

This tutorial is designed to provide the basic steps for creating DEMs from raw ASTER imagery, using MMASTER.

If you are planning to run this process on many DEMs, it might be worth having a look at scripts such as
:doc:`bash/process_mmaster.sh`, or :doc:`bash/RunMicMacAster_batch.sh`, which are designed to automatically
process many, many scenes at a time.

And of course, you are always free/encouraged to write your own scripts as needed.

Downloading data
################

ASTER data can most easily obtained from `NASA Earthdata <https://search.earthdata.nasa.gov/>`_. Be sure to
search the ASTER L1A Reconstructed Unprocessed Instrument Data, not the terrain-corrected data.

Once you have found a suitable number of scenes, be sure to order the **Geotiff**-formatted data.

After some time, you'll get an e-mail containing links to download your data. At this point, you're ready
to run MMASTER.


Running MMASTER to process ASTER DEMs
#####################################

If you have downloaded one or more **strips** (i.e., more than one ASTER scene that were acquired continuously), you
can use :doc:`python/scripts/sort_aster_strips` to sort these scenes into strips of typically no more than three
individual scenes.

From there, you can use :doc:`bash/WorkFlowASTER.sh` to run MMASTER and extract a DEM from the ASTER
imagery:
::

    WorkFlowASTER.sh -s SCENENAME -z "utm zone +north/south" -a -i 2

This will run the MMASTER workflows on a folder called SCENENAME (usually of the form AST_L1A_003MMDDYYYHHMMSS),
with outputs in UTM ZONE N/S. It will use version 2 of the fitting routine (more details `here </>`_), and it will
create track angle maps to be used in the bias removal stages. Support is also included for Polar Stereographic
coordinate systems; check :doc:`bash/WorkFlowASTER.sh` for more details, and for other command-line options.

Depending on the number of scenes, this may take some time (on our 80-core processing server at UiO it's about
1.5 hours per 3-scene strip; around 20-30 minutes for an individual scene). As stated in :doc:`setup`, it will
likely take significantly longer on most personal laptops, and thus we don't really recommend it.

As detailed in `Girod et al (2017) <https://www.mdpi.com/2072-4292/9/7/704/>`_, this process will
remove a significant portion of the cross-track bias in the DEM, as well as improve the matching, but
some cross-track bias will remain, as well as all of the along-track bias. In order to remove the remaining
biases, we have to use **pymmaster**.

Using pymmaster to remove remaining biases
##########################################

To do...