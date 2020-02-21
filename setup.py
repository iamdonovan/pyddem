from setuptools import setup

setup(name='pyddem',
      version='0.1',
      description='',
      url='https://github.com/iamdonovan/pyddem',
      author='Romain Hugonnet and Bob McNabb',
      author_email='robertmcnabb@gmail.com',
      license='GPL-3.0',
      packages=['pyddem'],
      install_requires=['numpy', 'scipy', 'matplotlib', 'fiona', 'pyvips',
                        'shapely', 'opencv-python', 'pandas', 'geopandas',
                        'scikit-image', 'gdal', 'h5py', 'pyproj', 'descartes',
                        'pybob>=0.24', 'netCDF4', 'xarray', 'numba', 'scikit-learn'],
      scripts=['bin/create_mmaster_stack.py', 'bin/sort_aster_strips.py','bin/sort_l1a.py',
               'bin/read_email_l1a.py', 'bin/fit_stack.py', 'bin/check_wget_l1a.py'],
      zip_safe=False)
