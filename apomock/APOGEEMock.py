# ----------------------------------------------------------------------------
#
# TITLE - APOGEEMock.py
# AUTHOR - James Lane
# CONTENTS
#
# ----------------------------------------------------------------------------

### Docstrings and metadata:
'''
'''
__author__ = "James Lane"

### Imports
import numpy as np
from galpy import potential
from galpy import orbit
from galpy.util import config,_rotate_to_arbitrary_vector
from astropy import units as apu
import astropy.coordinates
import scipy.interpolate
import scipy.integrate
import healpy.pixelfunc
import mwdust
from apomock.util.util import chabrier_imf,kroupa_imf,_xi_to_r,_r_to_xi,_cimf,Z2FEH

# ----------------------------------------------------------------------------

_DEGTORAD = np.pi/180.
_parsec_1_2_iso_keys = {'mass_initial':'Mini',
                        'z_initial':'Zini',
                        'log_age':'logAge',
                        'jmag':'Jmag',
                        'hmag':'Hmag',
                        'ksmag':'Ksmag',
                        'logg':'logg',
                        'logteff':'logTe'
                        }

class APOGEEMock:
    '''APOGEEMock:
    
    Class for mock data
    '''
    def __init__(self,denspot,ro=None,vo=None,zo=0.):
        '''__init__:
        
        Instantiate an APOGEEMock class. 
        
        Args:
            denspot (galpy.potential.Potential) - density potential, must 
                be spherically symmmetric
            ro,vo (float or astropy unit, optional) - galpy scale parameters.
                If None will try to set to denspot scale parameters if possible, 
                and galpy defaults if not. If not astropy unit then must be 
                in kpc and km/s. If ro,vo are supplied and denspot has 
                ro,vo set they should be equal but this is not checked.
            zo (float or astrop unit, optional) - Height of Sun above the 
                galactic disk.
        
        Returns:
            None
        ''' 
        
        # Density profile
        assert isinstance(denspot,potential.Potential),\
            'denspot must be galpy potential'
        self._denspot = denspot
        potential.turn_physical_off(self._denspot)
        
        # Get scale parameters
        if ro is None:
            try:
                self._ro = denspot._ro
            except AttributeError:
                try:
                    self._ro = denspot[0]._ro
                except (AttributeError,TypeError): # galpy defaults
                    self._ro = config.__config__.getfloat('normalization','ro')
        elif isinstance(ro,apu.quantity.Quantity):
            self._ro = ro.to(apu.kpc).value
        else: # Assume kpc
            self._ro = ro

        if vo is None:
            try:
                self._vo = denspot._vo
            except AttributeError:
                try:
                    self._vo = denspot[0]._vo
                except (AttributeError,TypeError): # galpy defaults
                    self._vo = config.__config__.getfloat('normalization','vo')
        elif isinstance(vo,apu.quantity.Quantity):
            self._vo = vo.to(apu.km/apu.s).value
        else: # Assume km/s
            self._vo = vo
        
        return None
    #def
    
    # Isochrone initialization 
    def load_isochrone(self,iso,iso_keys=_parsec_1_2_iso_keys):
        '''load_isochrone:
        
        Supply an isochrone for the mock. The isochrone should be a numpy 
        structured array. The dictionary iso_keys links the relevent keys 
        for the isochrone array with this common set of keys used by the code:
        
        The following are required for core functionality
        'mass_initial' - Initial mass of each point in the isochrone
        'jmag' - J-band magnitude
        'hmag' - H-band magnitude
        'ksmag' - Ks-band magnitude
        
        The following are only required to make a mock APOGEE allstar file
        'z_initial' - Initial metal fraction (z) of each point in the isochrone
        'logg' - Log base 10 surface gravity
        'logteff' - Log base 10 effective surface temperature
        
        So for example if the initial mass in the isochrone is accessed 
        by calling iso['Mini'], then one element of iso_keys should be 
        {...,'mass_initial':'Mini',...} and so on.
        
        Args:
            iso (numpy.ndarray) - Isochrone array
            iso_keys (dict) - Dictionary of keys for the isochrone
        
        Returns:
            None
        '''
        # Just initialization for now, perhaps more in the future
        self.iso = iso
        self.iso_keys = iso_keys
        return None
    #def
       
    def load_parsec_isochrone(self,filename,z,log_age,remove_wd_points=True,
                              iso_keys=_parsec_1_2_iso_keys):
        '''load_parsec_isochrone:

        Load an isochrone from a parsec grid

        Args:
            filename (string) - filename of the isochrone
            z (float) - metallicity (will use nearest)
            log_age (float) - log_age (will use nearest)
            remove_wd_points (bool, optional) - Remove any WD-like points from 
                the isochrone before matching [default True]
            iso_keys (dict, optional) - Dictionary that indexes a common set of keys 
                to the specific keys that query the isochrone 
                [default _parsec_1_2_iso_keys]
                
        Returns:
            None
        '''
        # Load
        iso = np.genfromtxt(filename, names=True, skip_header=13, comments='#') 
        
        # Find which z to use
        zs = np.unique(iso[_parsec_1_2_iso_keys['z_initial']])
        log_ages = np.unique(iso[_parsec_1_2_iso_keys['log_age']])
        if z in zs:
            z_choice = z
        else:
            z_choice = zs[np.argmin(np.abs(z-zs))]
            print('Using z='+str(z_choice))
        self._iso_z = z_choice

        # Find which log Age to use
        log_ages = np.unique(iso[_parsec_1_2_iso_keys['log_age']])
        if log_age in log_ages:
            log_age_choice = log_age
        else:
            log_age_choice = log_ages[np.argmin(np.abs(log_age-log_ages))]
            print('Using log age='+str(log_age_choice))
        self._iso_log_age = log_age_choice

        # Extract the isochrone
        iso_mask = (iso[_parsec_1_2_iso_keys['z_initial']] == z_choice) &\
                   (iso[_parsec_1_2_iso_keys['log_age']] == log_age_choice)
        iso = iso[iso_mask]

        # Remove any points that look like WDs, for parsec this is specifically
        # that logL = -9.999
        if remove_wd_points:
            iso_wd_mask = iso['logL']>-9
            iso = iso[iso_wd_mask]
        
        self.iso = iso
        self.iso_keys = _parsec_1_2_iso_keys
    
    # Mass sampling
    def sample_masses(self,m_tot,imf_type='chabrier',m_min=None,m_max=None,
                      force_resample=False):
        '''sample_masses:
        
        Draw mass samples from an IMF. 
        
        Args:
            m_tot (float) - Total mass worth of stars to sample in Msun
            imf_type (string, optional) - IMF type, either chabrier or kroupa.
                Default is chabrier
            m_min (float, optional) - minimum sample mass bound for the IMF in 
                Msun. If not supplied will be set to minimum mass in isochrone.
            m_max (float, optional) - maximum sample mass bound for the IMF in 
                Msun. If not supplied will be set to maximum mass in isochrone.
            force_resample (bool, optional) - Force a re-sample of masses, 
                overwriting existing masses [default False]
        '''
        if hasattr(self,'masses') and not force_resample:
            raise RuntimeError('Masses have already been sampled!')
        
        # Set the total mass
        if isinstance(m_tot,apu.quantity.Quantity):
            self._m_tot = m_tot.to(apu.M_sun).value
        else:
            self._m_tot = m_tot
        
        # Set the minimum and maximum mass, add a small buffer if reading 
        # from the isochrone
        if m_min is None:
            self._m_min = np.min(self.iso[self.iso_keys['mass_initial']])-0.01                
        else:
            if isinstance(m_min,apu.quantity.Quantity):
                self._m_min = m_min.to(apu.M_sun).value
            else:
                self._m_min = m_min
        if m_max is None:
            self._m_max = np.max(self.iso[self.iso_keys['mass_initial']])+0.01
        else:
            if isinstance(m_max,apu.quantity.Quantity):
                self._m_max = m_max.to(apu.M_sun).value
            else:
                self._m_max = m_max
        
        assert imf_type in ['chabrier','kroupa'],\
            'Only Chabrier and Kroupa IMFs currently supported'
        self._imf_type = imf_type
        
        # Make the icimf interpolator
        if self._imf_type == 'chabrier':
            imf_func = chabrier_imf
        elif self._imf_type == 'kroupa':
            imf_func = kroupa_imf
        icimf_interp = self._make_icimf_interpolator(imf_func,self._m_min,
                                                     self._m_max)
        
        # Guess how many samples to draw based on the average mass
        ms_for_avg = np.arange(self._m_min,self._m_max,0.01)
        m_avg = np.average(ms_for_avg,weights=imf_func(ms_for_avg))
        n_samples_guess = int(self._m_tot/m_avg)
        
        # Draw the first round of samples
        ms = np.power(10,icimf_interp(np.random.random(n_samples_guess)))
        
        # Add more samples or take some away depending on the total sampled mass
        while np.sum(ms) < self._m_tot:
            n_samples_guess = int((self._m_tot-np.sum(ms))/m_avg)
            if n_samples_guess < 1: break
            ms = np.append(ms,np.power(10,
                icimf_interp(np.random.random(n_samples_guess))))
        if np.sum(ms) > self._m_tot:
            ms = ms[:np.where(np.cumsum(ms) > self._m_tot)[0][0]]
        
        self.masses = ms.astype('float16')
    #def
    
    def _make_icimf_interpolator(self,imf,m_min,m_max):
        '''_make_icimf_interpolator:
        
        Make interpolator for the inverse cumulative initial  mass function 
        which maps normalized (0 to 1) cumulative IMF onto mass 
        (m_min to m_max). Note that the interpolator maps onto log10(m).

        Args:
            imf (callable) - Initial mass function
            m_min (float) - minimum mass (must be > 0)
            m_max (float) - maximum mass (must be finite)

        Returns:
            icimf_interp (scipy.interpolate.InterpolatedUnivariateSpline) - 
                icimf interpolated spline
        '''
        assert m_min > 0 and np.isfinite(m_max), 'mass range out of bounds'
        ms = np.logspace(np.log10(m_min),np.log10(m_max),1000)
        cml_imf = np.array([_cimf(imf,m_min,m) for m in ms])
        cml_imf /= cml_imf[-1] # Normalize

        return scipy.interpolate.InterpolatedUnivariateSpline(cml_imf,
            np.log10(ms), k=3)
    #def
    
    def sample_positions(self,n=None,denspot=None,r_min=0.,r_max=np.inf,
                         scale=None,b=None,c=None,zvec=None,pa=None,alpha=None,
                         beta=None,gamma=None,force_resample=False):
        '''sample_positions:

        Draw position samples from the density profile. Number of samples drawn 
        defaults to the number of masses if already sampled.

        Distribution of position samples can be modified to be triaxial using the 
        parameters b (ratio of Y to X scale lengths) and c (ratio of Z to X scale 
        lengths). 

        Distribution of position samples can also be rotated using either a 
        zvec + position angle scheme or yaw-pitch-roll scheme. For the former 
        the distribution is first rotated such that the original Z-vector (i.e. 
        the Z axis in galactocentric coordinates) is rotated to match zvec, and 
        then the distribution is rotated by pa. In the later the distribution 
        is rotated by a yaw, then pitch, and roll.

        Args:
            n (int) - Number of samples to draw [1]
            denspot (potential.Potential) - Potential representing density profile
            r_min (float) - Minimum radius to sample [0]
            r_max (float) - Maximum radius to sample [infinity]
            scale (float) - Density profile scale radius for mass sampling 
                interpolator (optional)
            b (float) - triaxial y/x scale ratio (optional)
            c (float) - triaxial z/x scale ratio (optional)
            zvec (list) - z-axis to align the new coordinate system (optional)
            pa (float) - Rotation about the transformed z-axis (optional)
            alpha (float) - Roll rotation about the x-axis  (optional)
            beta (float) - Pitch rotation about the transformed y-axis (optional)
            gamma (float) - Yaw rotation around twice-transformed z-axis (optional)
            return_orbits (bool) - Return the orbits [False]
            force_resample (bool) - Force a re-draw of masses, overwriting
                existing masses [False]

        Returns:
            None, position samples are saved as a galpy.orbit.Orbit object, 
                which can be accessed using .orbs attribute
        '''
        if hasattr(self,'orbs') and not force_resample:
            raise RuntimeError('Positions have already been sampled!')
        
        # The number of samples will be the number of masses
        if n is None:
            if hasattr(self,'masses'):
                n = len(self.masses)
            else:
                n = 1
        
        if denspot is None:
            denspot = self._denspot

        # Try and set the scale parameter
        if scale is None:
            try:
                self._scale = denspot._scale
            except AttributeError:
                try:
                    self._scale = denspot[0]._scale
                except (AttributeError,TypeError):
                    self._scale = 1.
        elif isinstance(scale,apu.quantity.Quantity):
            self._scale = scale.to(apu.kpc).value/self._ro
        else:
            self._scale = scale
        ##fi
        
        if isinstance(r_min,apu.quantity.Quantity):
            self._r_min = r_min.to(apu.kpc).value/self._ro
        else:
            self._r_min = r_min
        
        if isinstance(r_max,apu.quantity.Quantity):
            self._r_max = r_max.to(apu.kpc).value/self._ro
        else:
            self._r_max = r_max

        # Draw radial and angular samples
        r_samples = self._sample_r(denspot,n,self._r_min,self._r_max,
                                   a=self._scale)
        phi_samples,theta_samples = self._sample_position_angles(n=n)
        R_samples = r_samples*np.sin(theta_samples)
        z_samples = r_samples*np.cos(theta_samples)

        # apply triaxial scalings and a rotation if set
        if b is not None or c is not None:
            if c is None: c = 1.
            if b is None: b = 1.
            self._b = b
            self._c = c
            x_samples = R_samples*np.cos(phi_samples)
            y_samples = R_samples*np.sin(phi_samples)
            y_samples *= self._b
            z_samples *= self._c
            # Prioritize zvec transformation
            if zvec is not None or pa is not None:
                if zvec is None: zvec = [0.,0.,1.]
                if pa is None: pa = 0.
                self._zvec = zvec
                self._pa = pa
                x_samples,y_samples,z_samples = self._transform_zvecpa(
                    x_samples, y_samples, z_samples, zvec, pa)
            elif alpha is not None or beta is not None or gamma is not None:
                if alpha is None: alpha = 0.
                if beta is None: beta = 0.
                if gamma is None: gamma = 0.
                self._alpha = alpha
                self._beta = beta
                self._gamma = gamma
                x_samples,y_samples,z_samples = self._transform_alpha_beta_gamma(
                    x_samples, y_samples, z_samples, alpha, beta, gamma)
            R_samples = np.sqrt(x_samples**2.+y_samples**2.)
            phi_samples = np.arctan2(y_samples,x_samples)

        # Make into orbits
        orbs = orbit.Orbit(vxvv=np.array([R_samples,np.zeros(n),np.zeros(n),
            z_samples,np.zeros(n),phi_samples]).T,ro=self._ro,vo=self._ro)
        self.orbs = orbs

    def _sample_r(self,denspot,n,r_min,r_max,a=1.):
        '''_sample_r:

        Draw radial position samples. Note the function interpolates the 
        normalized iCMF onto the variable xi, defined as:

        .. math:: \\xi = \\frac{r/a-1}{r/a+1}

        so that xi is in the range [-1,1], which corresponds to an r range of 
        [0,infinity)

        Args:
            denspot (galpy.potential.Potential) - galpy potential representing
                the density profile. Must be spherical
            n (int) - Number of samples
            r_min (float) - Minimum radius to sample positions
            r_max (float) - Maximum radius to sample positions
            a (float) - Scale radius for interpolator

        Returns:
            r_samples (np.ndarray) - Radial position samples
        '''
        # First make the icmf interpolator
        icmf_xi_interp = self._make_icmf_xi_interpolator(denspot,r_min,
            r_max,a=a)

        # Now draw samples
        icmf_samples = np.random.uniform(size=int(n))
        xi_samples = icmf_xi_interp(icmf_samples)
        return _xi_to_r(xi_samples,a=a)
    
    def _make_icmf_xi_interpolator(self,denspot,r_min,r_max,a=1.):
        '''_make_icmf_xi_interpolator:

        Create the interpolator object which maps the iCMF onto variable xi.
        Note - the function interpolates the normalized CMF onto the variable 
        xi defined as:

        .. math:: \\xi = \\frac{r-1}{r+1}

        so that xi is in the range [-1,1], which corresponds to an r range of 
        [0,infinity)
        
        Note - must use self.xi_to_r() on any output of interpolator

        Args:

        Returns
            icmf_xi_interpolator
        '''
        xi_min= _r_to_xi(r_min,a=a)
        xi_max= _r_to_xi(r_max,a=a)
        xis= np.arange(xi_min,xi_max,1e-4)
        rs= _xi_to_r(xis,a=a)

        try:
            ms = potential.mass(denspot,rs,use_physical=False)
        except (AttributeError,TypeError):
            ms = np.array([potential.mass(denspot,r,use_physical=False)\
                           for r in rs])
        mnorm = potential.mass(denspot,r_max,use_physical=False)

        if r_min > 0:
            ms -= potential.mass(denspot,r_min,use_physical=False)
            mnorm -= potential.mass(denspot,r_min,use_physical=False)
        ms /= mnorm

        # Add total mass point
        if np.isinf(r_max):
            xis= np.append(xis,1)
            ms= np.append(ms,1)
        return scipy.interpolate.InterpolatedUnivariateSpline(ms,xis,k=3)
    
    def _sample_position_angles(self,n):
        '''_sample_position_angles:

        Draw galactocentric, spherical angle samples.

        Args:
            n (int) - Number of samples

        Returns:
            phi_samples (np.ndarray) - Spherical azimuth
            theta_samples (np.ndarray) - Spherical polar angle
        '''
        phi_samples= np.random.uniform(size=n)*2*np.pi
        theta_samples= np.arccos(1.-2*np.random.uniform(size=n))
        return phi_samples,theta_samples
    
    def _transform_zvecpa(self,x,y,z,zvec,pa):
        '''_transform_zvecpa:

        Transform coordinates using the axis-angle method. First align the
        z-axis of the coordinate system with a vector (zvec) and then rotate 
        about the new z-axis by an angle (pa).

        Args:
            x,y,z (array) - Coordinates
            zvec (list) - z-axis to align the new coordinate system
            pa (float) - Rotation about the transformed z-axis

        Returns:
            x_rot,y_rot,z_rot (array) - Rotated coordinates 
        '''
        pa_rot = np.array([[ np.cos(pa), np.sin(pa), 0.],
                           [-np.sin(pa), np.cos(pa), 0.],
                           [0.         , 0.        , 1.]])

        zvec /= np.sqrt(np.sum(zvec**2.))
        zvec_rot = _rotate_to_arbitrary_vector(np.array([[0.,0.,1.]]),
                                               zvec,inv=True)[0]
        R = np.dot(pa_rot,zvec_rot)

        xyz = np.squeeze(np.dstack([x,y,z]))
        if np.ndim(xyz) == 1:
            xyz_rot = np.dot(R, xyz)
            x_rot,y_rot,z_rot = xyz_rot[0],xyz_rot[1],xyz_rot[2]
        else:
            xyz_rot = np.einsum('ij,aj->ai', R, xyz)
            x_rot,y_rot,z_rot = xyz_rot[:,0],xyz_rot[:,1],xyz_rot[:,2]
        return x_rot,y_rot,z_rot

    def _transform_alpha_beta_gamma(self,x,y,z,alpha,beta,gamma):
        '''_transform_alpha_beta_gamma:

        Transform x,y,z coordinates by a yaw-pitch-roll transformation.

        Args:
            x,y,z (array) - Coordinates
            alpha (float) - Roll rotation about the x-axis 
            beta (float) - Pitch rotation about the transformed y-axis
            gamma (float) - Yaw rotation around twice-transformed z-axis

        Returns:
            x_rot,y_rot,z_rot (array) - Rotated coordinates 
        '''
        # Roll matrix
        Rx = np.zeros([3,3])
        Rx[0,0] = 1
        Rx[1]   = [0           , np.cos(alpha), -np.sin(alpha)]
        Rx[2]   = [0           , np.sin(alpha), np.cos(alpha)]
        # Pitch matrix
        Ry = np.zeros([3,3])
        Ry[0]   = [np.cos(beta), 0            , np.sin(beta)]
        Ry[1,1] = 1
        Ry[2]   = [-np.sin(beta), 0, np.cos(beta)]
        # Yaw matrix
        Rz = np.zeros([3,3])
        Rz[0]   = [np.cos(gamma), -np.sin(gamma), 0]
        Rz[1]   = [np.sin(gamma), np.cos(gamma), 0]
        Rz[2,2] = 1
        R = np.matmul(Rx,np.matmul(Ry,Rz))

        xyz = np.squeeze(np.dstack([x,y,z]))
        if np.ndim(xyz) == 1:
            xyz_rot = np.dot(R, xyz)
            x_rot,y_rot,z_rot = xyz_rot[0],xyz_rot[1],xyz_rot[2]
        else:
            xyz_rot = np.einsum('ij,aj->ai', R, xyz)
            x_rot,y_rot,z_rot = xyz_rot[:,0],xyz_rot[:,1],xyz_rot[:,2]
        return x_rot,y_rot,z_rot

    # Selection function application
    def apply_selection_function(self,aposf,dmap,iso=None,iso_keys=None,
                                 orbs=None,ms=None,force_reapply=False,
                                 print_stats=False):
        '''apply_selection_function:

        Apply the APOGEE selection function to sampled data. The order of
        operations is:
        1. Match isochrone to the samples based on initial mass
        2. Remove samples with magnitudes fainter than the faintest 
            APOGEE Hmax
        3. Remove samples which lie outside the APOGEE footprint
        4. Remove samples with magnitudes fainter than the APOGEE Hmax 
            one a field-by-field basis
        5. Calculate H-band extinction and then apply the APOGEE 
            selection function
        
        Args:
            aposf (apogee.select.*) - APOGEE selection function
            dmap (mwdust.DustMap3D) - Dust map
            iso (np.array) - Isochrone
            iso_keys (dict) - Isochrone key dictionary, see load_isochrone()
            orbs (galpy.orbit.Orbit) - Orbits representing the samples
            ms (np.array) - Masses of the samples
            force_reapply (bool) - force a re-application of the selection 
                function
            
        Returns:
            None, .orbs and .masses attributes are updated to hold the 
                samples which survive application of the APOGEE selection
                function. The .locid attribute holds APOGEE field location 
                IDs of samples. The .iso_match_indx attribute holds 
                indices of the isochrone which were matched to the samples.
        '''
        if all(hasattr(self,attr) for attr in ['iso_match_indx','locid']) \
            and not force_reapply:
            raise RuntimeError('Selection function has already been applied!')
            
        if iso is None or iso_keys is None:
            iso = self.iso
            iso_keys = self.iso_keys
        if orbs is None:
            orbs = self.orbs
        if ms is None:
            ms = self.masses
        ncur = len(ms)
        
        # Get some information about APOGEE - place these somewhere more 
        # appropriate so they make sense
        nspec = np.nansum(aposf._nspec_short,axis=1) +\
                np.nansum(aposf._nspec_medium,axis=1) +\
                np.nansum(aposf._nspec_long,axis=1)
        good_nspec_fields = np.where(nspec>=1.)[0]

        aposf_Hmax = np.dstack([aposf._short_hmax,
                                aposf._medium_hmax,
                                aposf._long_hmax])[0]

        # Match samples to isochrone entries based on initial mass
        ncur = len(ms)
        m_err = np.diff(iso[iso_keys['mass_initial']]).max()/2.+1e-4
        good_iso_match,iso_match_indx = self._match_isochrone_to_samples(
            iso=iso,ms=ms,m_err=m_err,iso_keys=iso_keys)
        # assert np.all(np.abs(ms[good_iso_match]-iso['Mini'][iso_match_indx]<=m_err))
        orbs = orbs[good_iso_match]
        ms = ms[good_iso_match]
        Hmag = iso['Hmag'][iso_match_indx]
        if print_stats:
            print(str(len(good_iso_match))+'/'+str(ncur)+\
                  ' samples have good matches in the isochrone')
            print('Kept '+str(round(100*len(good_iso_match)/ncur,2))+\
                  ' % of samples')
        ncur = len(ms)

        # Remove samples with apparent Hmag below faintest APOGEE Hmax
        dm = 5.*np.log10(orbs.dist().to(apu.pc).value)-5.
        where_good_Hmag1 = np.where(np.nanmax(aposf_Hmax) > (Hmag+dm) )[0]
        orbs = orbs[where_good_Hmag1]
        ms = ms[where_good_Hmag1]
        dm = dm[where_good_Hmag1]
        Hmag = Hmag[where_good_Hmag1]
        iso_match_indx = iso_match_indx[where_good_Hmag1]
        if print_stats:
            print(str(len(where_good_Hmag1))+'/'+str(ncur)+\
                  ' samples are bright enough to be observed')
            print('Kept '+str(round(100*len(where_good_Hmag1)/ncur,2))+\
                  ' % of samples')
        ncur = len(ms)

        # Remove samples that lie outside the APOGEE observational footprint
        fp_indx,locid = self._remove_samples_outside_footprint(orbs,aposf,
            good_nspec_fields)
        orbs = orbs[fp_indx]
        ms = ms[fp_indx]
        dm = dm[fp_indx]
        Hmag = Hmag[fp_indx]
        iso_match_indx = iso_match_indx[fp_indx]
        if print_stats:        
            print(str(len(fp_indx))+'/'+str(ncur)+' samples found within'+\
                  ' observational footprint')
            print('Kept '+str(round(100*len(fp_indx)/ncur,2))+\
                  ' % of samples')
        ncur = len(ms)

        # Remove samples with apparent Hmag below faintest Hmax on a
        # field-by-field basis
        field_Hmax = np.nanmax(aposf_Hmax, axis=1)
        locid_inds = np.where(locid.reshape(locid.size, 1) ==\
                              aposf._locations)[1]
        Hmax = field_Hmax[locid_inds]
        where_good_Hmag2 = np.where(Hmax > (Hmag+dm))[0]
        orbs = orbs[where_good_Hmag2]
        locid = locid[where_good_Hmag2]
        ms = ms[where_good_Hmag2]
        dm = dm[where_good_Hmag2]
        Hmag = Hmag[where_good_Hmag2]
        iso_match_indx = iso_match_indx[where_good_Hmag2]
        Jmag = iso['Jmag'][iso_match_indx]
        Ksmag = iso['Ksmag'][iso_match_indx]
        if print_stats:
            print(str(len(where_good_Hmag2))+'/'+str(ncur)+\
                      ' samples are bright enough to be observed')
            print('Kept '+str(round(100*len(where_good_Hmag2)/ncur,2))+\
                  ' % of samples')
        ncur = len(ms)

        # Get lbIndx for the dust map and compute AH
        lbIndx = self._get_lbIndx(orbs,dmap)
        AH = self._calculate_AH(dmap,lbIndx,dm)

        # Apply the selection function
        Hmag_app = Hmag + dm  + AH
        JK0 = Jmag - Ksmag
        sf_keep_indx = np.zeros(len(orbs),dtype=bool)
        for i in range(len(orbs)):
            sf_prob = aposf(locid[i],Hmag_app[i],JK0[i])
            sf_keep_indx[i] = sf_prob > np.random.random(size=1)[0] 
        if print_stats:
            print(str(np.sum(sf_keep_indx))+'/'+str(ncur)+\
                      ' samples survive the selection function')
            print('Kept '+str(round(100*np.sum(sf_keep_indx)/ncur,2))+\
                  ' % of samples')

        self.orbs = orbs[sf_keep_indx]
        self.locid = locid[sf_keep_indx]
        self.masses = ms[sf_keep_indx]
        self.iso_match_indx = iso_match_indx[sf_keep_indx]
    #def
    
    def _match_isochrone_to_samples(self,iso,ms,m_err,iso_keys):
        '''_match_isochrone_to_samples:

        Match the samples to entries in an isochrone according to initial mass

        iso_keys must accept the following keys:
        'Mini' -> initial mass key

        Args:
            iso (array) - isochrone array
            ms (array) - sample masses
            m_err (float) - Maximum difference in mass between sample and 
                isochrone for successful match
            iso_keys (dict) - Dictionary of keys for accessing the isochrone 
                properties, accessible via a common set of strings (see above)

        Returns:
            good_match (array) - Indices of ms which found matches in the 
                isochrone array within m_err tolerance
            match_indx (array) - array of matches, length len(good_match), 
                indexing ms into iso
        '''
        # Access initial mass
        m0 = iso[iso_keys['mass_initial']]

        # Search the sorted isochrone for nearest neighbors
        m0_argsort = np.argsort(m0)
        m0_sorted = m0[m0_argsort]
        m0_mids = m0_sorted[1:] - np.diff(m0_sorted.astype('f'))/2
        idx = np.searchsorted(m0_mids, ms)
        cand_indx = m0_argsort[idx]
        residual = ms - m0_sorted[cand_indx]
        
        # Pick the masses which lie within the mass range of the isochrone
        # and are separated by the mass error
        good_match = np.where( (np.abs(residual) < m_err) &\
                               (ms < m0[-1]+1e-4) &\
                               (ms > m0[0]-1e-4)
                              )[0]
        match_indx = np.argsort(m0_argsort)[cand_indx[good_match]]
        np.all(np.abs(ms[good_match]-m0[match_indx]) <= m_err)

        return good_match,match_indx
    #def
    
    def _remove_samples_outside_footprint(self,orbs,aposf,field_indx=None):
        '''_remove_samples_outside_footprint:

        Remove stellar samples from outside the APOGEE observational footprint.
        Each plate has a variable field of view, and an inner 'hole' of 
        5 arcminutes. Using field_indx allows for selecting only a subset of the 
        available fields to use.

        Args:
            orbs (array) - Orbits representing samples
            aposf (array) - APOGEE selection function
            field_indx (array) - Indices of fields to consider

        Returns:
            fp_indx (np.array) - Index of samples that lie within the 
                observational footprint
            fp_locid (np.array) - Location IDs of field each sample lies within
        '''
        # Account for field_indx, fields we want to consider
        if field_indx is None:
            field_indx = np.arange(0,len(aposf._apogeeFields),dtype=int)
        ##fi

        # field center coordinates, location IDs, radii
        glon = aposf._apogeeField['GLON'][field_indx]
        glat = aposf._apogeeField['GLAT'][field_indx]
        locids = aposf._locations[field_indx]
        radii = np.zeros(len(field_indx))
        for i in range(len(locids)):
            radii[i] = aposf.radius(locids[i])

        # Make SkyCoord objects
        aposf_sc = astropy.coordinates.SkyCoord(frame='galactic', 
                           l=glon*apu.deg, b=glat*apu.deg)
        orbs_sc = astropy.coordinates.SkyCoord(frame='galactic', 
                           l=orbs.ll(use_physical=True), 
                           b=orbs.bb(use_physical=True))

        # First nearest-neighbor match
        indx,sep,_ = orbs_sc.match_to_catalog_sky(aposf_sc)
        indx_radii = radii[indx]
        indx_locid = locids[indx]
        fp_indx = np.where(np.logical_and(sep < indx_radii*apu.deg,
                                          sep > 5.5*apu.arcmin))[0]
        fp_locid = indx_locid[fp_indx]

        # Second nearest-neighbor match for samples inside plate central holes
        where_in_hole = np.where(sep < 5.5*apu.arcmin)[0]
        indx2,sep2,_ = orbs_sc[where_in_hole].match_to_catalog_sky(aposf_sc,
                                                                   nthneighbor=2)
        indx2_radii = radii[indx2]
        indx2_locid = locids[indx2]
        fp_indx2 = np.where(np.logical_and(sep2 < indx2_radii*apu.deg,
                                           sep2 > 5.5*apu.arcmin))[0]
        if len(fp_indx2) > 0:
            fp_indx = np.append(fp_indx,where_in_hole[fp_indx2])
            fp_locid = np.append(fp_locid,indx2_locid[fp_indx2])

        return fp_indx,fp_locid
    
    def _get_lbIndx(self,orbs,dmap):
        '''_get_lbIndx:
        
        Get the index of each sample into the dustmap. The dustmap is 
        arranged as a hierarchical healpix map, with cells at multiple 
        resolutions. Find the index of each sample in this structure
        
        Args:
            orbs (orbit.Orbit) - phase space samples
            dmap (mwdust.DustMap3D) - Dust map
            
        Returns:
            lbindx (np.array) - dustmap indices of samples
        '''
        gl = orbs.ll(use_physical=True).value
        gb = orbs.bb(use_physical=True).value
        dist = np.atleast_2d(orbs.dist(use_physical=True).to(apu.kpc).value).T
        # Prepare arrays to hold healpix information for samples
        dmap_nsides = np.array(dmap._nsides)
        pix_arr = np.zeros((len(orbs),len(dmap_nsides)))
        nside_arr = np.repeat(dmap_nsides[:,np.newaxis],len(orbs),axis=1).T
        # Calculate healpix pixels for samples
        for i in range(len(dmap_nsides)):
            pix_arr[:,i] = healpy.pixelfunc.ang2pix(dmap_nsides[i],
                                                    (90.-gb)*_DEGTORAD,
                                                    gl*_DEGTORAD, nest=True)
        # Calculate healpix u for dust map and samples
        dmap_hpu = (dmap._pix_info['healpix_index'] +\
                    4*dmap._pix_info['nside']**2.).astype(int)
        hpu = (pix_arr + 4*nside_arr**2).astype(int)
        # Use searchsorted to match sample u to dust map u
        dmap_hpu_argsort = np.argsort(dmap_hpu)
        dmap_hpu_sorted = dmap_hpu[dmap_hpu_argsort]
        hpu_indx_sorted = np.searchsorted(dmap_hpu_sorted,hpu)
        hpu_indx = np.take(dmap_hpu_argsort, hpu_indx_sorted, mode="clip")
        hpu_mask = dmap_hpu[hpu_indx] != hpu
        hpu_ma = np.ma.array(hpu_indx, mask=hpu_mask)
        lbIndx = hpu_ma.data[~hpu_ma.mask]
        return lbIndx
    
    def _calculate_AH(self,dmap,lbIndx,dm):
        '''_calculate_AH:
        
        Calculate H-band extinction
        
        Args:
            dmap (mwdust.DustMap3D) - Dust map
            lbIndx (np.array) - dustmap indices of samples
            dm (np.array) - Distance moduli of samples
        
        Returns:
            AH (np.array) - H-band extinction
        '''
        unique_lbIndx = np.unique(lbIndx).astype(int)
        AH = np.zeros(len(lbIndx))
        for i in range(len(unique_lbIndx)):
            where_unique = np.where(lbIndx == unique_lbIndx[i])[0]
            # Get the dust map interpolation data for this lbIndx
            dmap_interp_data = scipy.interpolate.InterpolatedUnivariateSpline(
                dmap._distmods, dmap._best_fit[unique_lbIndx[i]], k=dmap._interpk)
            # Calcualate AH
            eBV_to_AH = mwdust.util.extCurves.aebv(dmap._filter,sf10=dmap._sf10)
            AH[where_unique] = dmap_interp_data(dm[where_unique])*eBV_to_AH
        return AH
    
    def make_allstar(self):
        '''make_allstar:

        Make a numpy structured array from the samples in the mock that mimics
        the APOGEE allstar file. Only includes fields necessary for density 
        modelling: [Fe/H], Location ID, Teff, Logg

        Args:
            iso (array) - Isochrone
            iso_match_indx (array) - Indices which match samples to isochrone points
            locid (array) - location IDs of APOGEE field where each sample lies
            fe_h (array) - [Fe/H] abundance of samples
        '''
        assert all(hasattr(self,attr) for attr in \
                   ['iso','iso_match_indx','locid']), 'Must run apply_selection_function'
        iso_match = self.iso[self.iso_match_indx]
        atype = np.dtype([('LOCATION_ID', 'i4'),
                         ('LOGG', 'f4'),
                         ('TEFF', 'f4'),
                         ('FE_H', 'f4')
                         ])
        allstar = np.empty(len(self.iso_match_indx), dtype=atype)
        allstar['LOCATION_ID'] = self.locid
        allstar['LOGG'] = iso_match[self.iso_keys['logg']]
        allstar['TEFF'] = iso_match[self.iso_keys['logteff']]
        allstar['FE_H'] = Z2FEH(iso_match[self.iso_keys['z_initial']])
        return allstar
    #def