# apomock
Create mock APOGEE data

## Installation

Standard python setup.py build/install

`sudo python setup.py install`

### Dependancies

This package required `numpy`, `scipy`, `astropy`, `galpy`, and `healpy`. All of these 
are installed via setup.py. Additionally, the package requires `apogee` (available [here](https://github.com/jobovy/apogee)) and `mwdust` (available [here](https://github.com/jobovy/mwdust)) which are not automatically installed. This program requires the dust maps installed with `mwdust`. The use of the `apogee` package is limited to selection functions, and actual APOGEE data is not required.

## Generating mock APOGEE data

The program generates mock data in a 3-step process
1. Generate mass samples using an IMF
2. Generate position samples using a `galpy` density profile
3. Apply APOGEE observational model

Application of the APOGEE observational model:
- matching samples to an isochrone based on stellar mass
- removing samples lying outside the APOGEE observational footprint
- Determining H-band extinction using `mwdust`
- Apply the APOGEE selection function using redenned H magnitudes and unredenned J-Ks color

### Quick start

Mock generation requires an isochrone to provide H, J, and Ks magnitudes. A 
sample grid of PARSEC v1.2 isochrones can be downloaded [here](https://www.astro.utoronto.ca/~lane/share/apomock/data/). This grid has metal fraction 0.0001 <= z <= 0.0060 with spacing 0.0001, and 10 Gyr <= age <= 14 Gyr with spacing 0.5 Gyr.

To begin import dependancies:
```import apomock
from galpy import potential
from apogee import select
from astropy import units as apu
import mwdust
```

We will create mock data corresponding to a power law density profile with slope `alpha=2`
```denspot = potential.PowerSphericalPotential(alpha=2.)
mock = apomock.APOGEEMock(denspot=denspot)
```

We will now load in the isochrone, if the sample PARSEC grid has been downloaded it can be initialized using `load_parsec_isochrone()`, otherwise use `load_isochrone()`. We will select metal fraction of 0.0010 and age of 11 Gyr
`mock.load_parsec_isochrone(filename=path_to_parsec_isochrone_grid,z=0.0010,log_age=np.log10(1.1e10))`

First sample masses from a Chabrier IMF, we'll make it `10^7` solar masses total. Astropy units are supported for most input.
`mock.sample_masses(m_tot=1e7*apu.M_sun)`

Now we draw position samples, we'll make the samples lie between galactocentric 
radius 2 kpc and 70 kpc.
`mock.sample_positions(r_min=2*apu.kpc, r_max=70*apu.kpc)`

Now we load in the dust map and define the APOGEE selection function. We'll use the most recent `Combined19` dust map, and we'll use the main APOGEE DR16 selection function (year 7). Calculation of the selection function is time consuming, it may be useful to save the selection function for future use.
```dmap = mwdust.Combined19(filter='2MASS H')
aposf = select.apogeeCombinedSelect(sample='main',year=7)
```

Finally apply the observational program
`mock.apply_selection_function(aposf, dmap)`

The phase-space properties of the mock are held in a `galpy.orbit.Orbit` instance at `mock.orbs`. The isochrone-based properties of the mock, such as magnitudes, surface gravity, effective temperature, and more, are accessed using `mock.iso[mock.iso_match_indx]`. The masses of the samples are accessed at `mock.masses`. The APOGEE Location IDs of each sample are accessed at `mock.locid`. 

A mock APOGEE allstar file can be generated using `mock.make_allstar()`.
