Installation and Setup
=======================

The following is a (non-exhaustive) set of instructions for getting setup to run MMASTER on your own machine. Note
that this can be an **extremely** computationally intensive process, so we don't really recommend trying to run this on your
personal laptop.

As this is a (non-exhaustive) set of instructions, it may not work 100% with your particular setup.
We are happy to try to provide guidance/support, **but we make no promises**.

Installing MicMac
#################

Detailed installation instructions for MicMac on multiple platforms can be found `here <https://micmac.ensg.eu/index.php/Install/>`_.

The basic idea is: clone the micmac git repository, then build the source code. Simple!

Optional: Preparing a python environment
########################################

If you like, you can set up a dedicated python environment for your MMASTER needs. This can be handy, in case any
packages required by pymmaster clash with packages in your default environment. Our personal preference is `conda <https://docs.conda.io/en/latest/>`_,
but your preferences may differ.

The git repository has a file, mmaster_environment.yml, which provides a working environment for pymmaster and conda.
Once you have conda installed, simply run:
::

    conda env create -f mmaster_environment.yml

This will create a new conda environment, called mmaster_environment, which will have all of the various python packages
necessary to run pymmaster. To activate the new environment, type:
::

    conda activate mmaster_environment

And you should be ready to go. Note that you will have to activate this environment any time you wish to run MMASTER
and pymmaster scripts and tools, if it is not already activated in your terminal.

Installing MMASTER and pymmaster
################################

Once you have MicMac installed and an (optional) python environment set up, you can install pymmaster.

First, clone the git repository:
::

    git clone https://github.com/luc-girod/MMASTER-workflows.git

Next, use **pip** to install the scripts and python modules:
::

    pip install -e MMASTER-workflows

Note: the *-e* allows you to make changes to the code (for example, from git updates or through your own tinkering),
that will then be updated within your python install. If you run *pip install* without this option, it will install
a static version of the package, and any changes/updates will have to be explictly re-installed.

Assuming that you haven't run into any errors, you should be set up. You can verify this by running:
::

    mmaster_bias_correction.py -h

From the command line (in a non-Windows environment; *Windows instructions coming soon-ish*).

You should see the following output (or something very similar):
::

    usage: mmaster_bias_correction.py [-h] [-s SLAVEDEM] [-a EXC_MASK]
                                  [-b INC_MASK] [-n NPROC] [-o OUTDIR] [-p]
                                  [-l]
                                  masterdem indir [indir ...]

    Run MMASTER post-processing bias corrections, given external elevation data.

    positional arguments:
        masterdem             path to master DEM/elevations to be used for bias
                              correction
        indir                 directory/directories where final, georeferenced
                              images are located.

    optional arguments:
        -h, --help            show this help message and exit
        -s SLAVEDEM, --slavedem SLAVEDEM
                            (optional) name of DEM to correct. By default,
                            mmaster_bias_correction.py looks for MMASTER DEMs of
                            the form AST_L1A_..._Z.tif
        -a EXC_MASK, --exc_mask EXC_MASK
                            exclusion mask. Areas inside of this shapefile (i.e.,
                            glaciers) will not be used for coregistration [None]
        -b INC_MASK, --inc_mask INC_MASK
                            inclusion mask. Areas outside of this mask (i.e.,
                            water) will not be used for coregistration. [None]
        -n NPROC, --nproc NPROC
                            number of processors to use [1].
        -o OUTDIR, --outdir OUTDIR
                            directory to output files to (creates if not already
                            present). [.]
        -p, --points          process assuming that master DEM is point elevations
                            [False].
        -l, --log             write output to a log file rather than printing to the
                            screen [False].


If you don't see this, feel free to ask for help by sending an e-mail, though it can also be helpful to google around
for some solutions first. If you do send us an e-mail, be sure to include the specific error messages that you receive.
Screenshots are also helpful.

Good luck!