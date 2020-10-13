from setuptools import setup

setup(name='pyddem',
      version='0.1',
      description='pyddem is a python package for processing time series of elevation measurements',
      url='https://github.com/iamdonovan/pyddem',
      author='Romain Hugonnet and Bob McNabb',
      author_email='robertmcnabb@gmail.com',
      license='GPL-3.0',
      packages=['pyddem'],
      install_requires=['dask', 'fiona', 'gdal', 'geopandas', 'h5py', 'llc',
                        'matplotlib', 'netCDF4', 'numba', 'numpy', 'pandas', 'pybob>=0.25', 'pymmaster>=0.1',
                        'pyproj', 'scikit-image', 'scikit-learn', 'scipy', 'skgstat', 'xarray'],
      scripts=['bin/fit_gp_monthly_aster.py', 'bin/fit_stack.py', 'bin/stack_dems.py',
               'bin/stack_per_1deg_tiles_aster.py'],
      zip_safe=False)
