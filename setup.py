import setuptools
import sys
import os
import subprocess

_include_package_data = False
_package_data={'apomock':['data/2mass-spitzer-wise-0.0001-z-0.0060-1e10-age-1.4e10.dat']}
_PARSEC_ISO_GRID_NAME = '2mass-spitzer-wise-0.0001-z-0.0060-1e10-age-1.4e10.dat'
_PARSEC_ISO_GRID_URL = 'https://www.astro.utoronto.ca/~lane/share/apomock/data/'+_PARSEC_ISO_GRID_NAME
if '--no-downloads' not in sys.argv:
    print('\033[1m'+'Downloading PARSEC v1.2 isochrone grid'+'\033[0m')
    os.makedirs('./apomock/data', exist_ok=True)
    try:
        subprocess.check_call(['wget',_PARSEC_ISO_GRID_URL,'-O',os.getcwd()+\
                               '/apomock/data/'+_PARSEC_ISO_GRID_NAME])
	_include_package_data = True
    except subprocess.CalledProcessError:
	_include_package_data = False
        print('\033[1m'+'Downloading PARSEC v1.2 isochrone grid failed'+'\033[0m')

setuptools.setup(
    name="apomock",
    version="0.1",
    author="James Lane",
    author_email="lane@astro.utoronto.ca",
    description="Mock data for APOGEE",
    packages=setuptools.find_packages(include=['apomock','apomock.util']),
    include_package_data=_include_package_data,
    package_data=_package_data,
    url='http://github.com/jamesmlane/apomock',
    install_requires=['numpy','scipy','galpy','healpy','astropy']
)
