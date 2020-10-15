Fitting a time series of DEMs with pyddem
=========================================

Once you have :doc:`created <stack_dems>` a stack of DEMs, :mod:`pyddem.fit_tools` has a number of functions to fit
a time series of DEMs. The main function, which will take a stack of DEMs and produce a fitted time series,
is :func:`pyddem.fit_tools.fit_stack`, which can also be run using the command-line tool :doc:`pyddem/scripts/stack_dems`.

This assumes that you have first created a stack of DEMs following the instructions provided :doc:`here <stack_dems>`.

Running from a script
#####################
The following example shows how you can run :func:`pyddem.fit_tools.fit_stack` to fit a linear trend using
weighted least-squares  to a stack of DEMs. It will filter the raw elevations using a reference DEM and a spatial
filter, fit only pixels that fall within the provided land mask, and provide a 95% confidence interval to the fit,
using 2 cores. For more information on the various parameters, see :func:`pyddem.fit_tools.fit_stack`.
::
    from pyddem.fit_tools import fit_stack


    ref_dem = 'arcticdem_mosaic_100m_v3.0.tif'
    ref_dem_date = '2013-08-01'
    fn_landmask = '~/data/Alaska_Coastline.shp',

    fit_stack('my_stack.nc',
              fn_ref_dem=ref_dem,
              ref_dem_date=ref_dem_date,
              filt_ref='min_max',
              inc_mask=fn_landmask,
              nproc=2,
              method='wls',
              conf_filt_ls=0.95)

The resulting fit will be written to a netCDF file called **fit.nc**. **TODO**

Running from the command line
#############################
This example will do the same as above, but using the command-line tool :doc:`pyddem/scripts/fit_stack`.
::
    fit_stack.py my_stack.nc -ref_dem arcticdem_mosaic_100m_v3.0.tif -ref_dem_date 2013-08-01 -f min_max
        -inc_mask ~/data/Alaska_Coastline.shp -n 2 -m wls -ci 0.95

That's it! The last thing to do is to open up the netCDF file and check the results. After that, you can use
:doc:`pyddem/modules/volint_tools` to calculate volume changes from your fitted elevation changes. Good luck!