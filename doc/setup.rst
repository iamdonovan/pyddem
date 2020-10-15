Installation and Setup
=======================

The following is a (non-exhaustive) set of instructions for getting setup to run pyddem on your own machine.

As this is a (non-exhaustive) set of instructions, it may not work 100% with your particular setup.
We are happy to try to provide guidance/support, **but we make no promises**.

Optional: Preparing a python environment
########################################

If you like, you can set up a dedicated python environment for your pyddem needs, or install into an existing environment.
Setting up a dedicated environment can be handy, in case any packages required by pymmaster clash with packages in your default environment. Our personal preference is `conda <https://docs.conda.io/en/latest/>`_,
but your preferences/experience may differ.

The git repository has a file, pyddem_environment.yml, which provides a working environment for pyddem and conda.
Once you have conda installed, simply run:
::

    conda env create -f pyddem_environment.yml

This will create a new conda environment, called pyddem_environment, which will have all of the various python packages
necessary to run pyddem_environment. To activate the new environment, type:
::

    conda activate pyddem_environment

And you should be ready to go. Note that you will have to activate this environment any time you wish to run pyddem, if it is not already activated in your terminal.

Installing from PyPI
################################

As of October 2020, pyddem v0.1 is available via the Python Package Index (PyPI). As such, it can be installed directly
using pip:
::
    pip install pyddem

This will install version 0.1 of the package. As time allows, we will continue to update the various functions, but this
will install a working version that you can use to stack and fit time series of DEMs, as well as perform various statistical
analysis on the DEMs and dDEMs.


Installing a development version
################################

If you would prefer to install a version that you can update yourself, you can do so as well.

First, clone the git repository:
::

    git clone https://github.com/iamdonovan/pyddem.git

Next, use **pip** to install the scripts and python modules:
::

    pip install -e pyddem

Note: the *-e* allows you to make changes to the code (for example, from git updates or through your own tinkering),
that will then be updated within your python install. If you run *pip install* without this option, it will install
a static version of the package, and any changes/updates will have to be explictly re-installed.

Checking the installation
################################

Assuming that you haven't run into any errors, you should be set up and ready to go. You can verify this by running:
::

    stack_dems.py -h

From the command line (in a non-Windows environment; *Windows instructions coming soon-ish*).

You should see the following output (or something very similar):
::
    usage: fit_stack.py [-h] [-b INC_MASK] [-n NPROC] [-t TIME_RANGE TIME_RANGE]
                        [-o OUTFILE] [-c]
                        stack

    Fit time series of elevation data using Gaussian Process.

    positional arguments:
      stack                 NetCDF file of stacked DEMs to fit.

    optional arguments:
      -h, --help            show this help message and exit
      -b INC_MASK, --inc_mask INC_MASK
                            inclusion mask. Areas outside of this mask (i.e.,
                            water) will be omitted from the fitting. [None]
      -n NPROC, --nproc NPROC
                            number of processors to use [1].
      -t TIME_RANGE TIME_RANGE, --time_range TIME_RANGE TIME_RANGE
                            Start and end dates to fit time series to (default is
                            read from input file).
      -o OUTFILE, --outfile OUTFILE
                            File to save results to. [fitted_stack.nc]
      -c, --clobber         Clobber existing outfile [False].


If you don't see this, feel free to ask for help by sending an e-mail, though it can also be helpful to google around
for some solutions first. If you do send us an e-mail, be sure to include the specific error messages that you receive.
Screenshots are also helpful.

Good luck!