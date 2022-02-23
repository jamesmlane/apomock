# ----------------------------------------------------------------------------
#
# TITLE - APOGEEMock.py
# AUTHOR - James Lane
#
# ----------------------------------------------------------------------------

'''APOGEEMock class tests'''

### Imports
from apomock import APOGEEMock
from apomock.util.util import chabrier_imf,kroupa_imf
import numpy as np
from astropy import units as apu
from galpy import potential,orbit
import scipy.integrate
import mwdust

_ro,_vo = 8.,220.

def test_initialization():
    '''test_class_initialization:
    
    Tests that the APOGEEMOCK class initializes with correct properties
    '''
    # denspot correct
    denspot = potential.HernquistPotential()
    mock = APOGEEMock(denspot=denspot)
    assert isinstance(mock._denspot, potential.HernquistPotential),\
        'denspot not set correctly'
    
    # ro/vo correct
    ros = [8.1,8.275]
    vos = [220.,231.7]
    for ro,vo in zip(ros,vos):
        mock = APOGEEMock(denspot=denspot,ro=ro,vo=vo)
        assert mock._ro == ro, 'ro not set correctly'
        assert mock._vo == vo, 'vo not set correctly'
    
    # ro/vo in astropy units correct
    ros = [8.2*apu.kpc, 8347.*apu.pc]
    vos = [224.*apu.km/apu.s, 2.3*apu.pc/apu.Myr]
    for ro,vo in zip(ros,vos):
        mock = APOGEEMock(denspot=denspot,ro=ro,vo=vo)
        assert mock._ro == ro.to(apu.kpc).value,\
            'ro in astropy units not set correctly'
        assert mock._vo == vo.to(apu.km/apu.s).value,\
            'vo in astropy units not set correctly'

    # ro/vo inheritance from denspot
    ros = [8.2*apu.kpc, 8347.*apu.pc, 8.275]
    vos = [224.*apu.km/apu.s, 2.3*apu.pc/apu.Myr, 231.7]
    for ro,vo in zip(ros,vos):
        denspot = potential.HernquistPotential(ro=ro,vo=vo)
        mock = APOGEEMock(denspot=denspot)
        assert mock._ro == denspot._ro,'ro not inherited from denspot'
        assert mock._vo == denspot._vo,'vo not inherited from denspot'

def test_sample_mass_distribution_parameters():
    '''test_sample_mass_distribution_parameters
    
    Test that the distribution of masses is bounded according to supplied 
    parameters
    '''
    imf_types = ['chabrier','kroupa']
    denspot = potential.HernquistPotential()
    mock = APOGEEMock(denspot=denspot)
    
    m_mins = [0.1,0.2]
    m_maxs = [0.8,0.9]
    m_tots = [1e3,1e4]
    for imf in imf_types:
        for m_min,m_max,m_tot in zip(m_mins,m_maxs,m_tots):
                mock.sample_masses(m_tot, imf_type=imf, m_min=m_min,
                                   m_max=m_max, force_resample=True)
                ms = mock.masses
                assert np.min(ms) > m_min, 'Masses sampled below m_min'
                assert np.max(ms) < m_max, 'Masses sampled above m_max'
                assert np.fabs(np.sum(ms)-m_tot) < m_max,\
                    'Total mass sampled does not match m_tot'
    
    m_mins = [0.1*apu.Msun,200.*apu.Mjup]
    m_maxs = [800.*apu.Mjup,0.9*apu.Msun]
    m_tots = [1e3*apu.Msun,1e7*apu.Mjup]
    for imf in imf_types:
        for m_min,m_max,m_tot in zip(m_mins,m_maxs,m_tots):
                mock.sample_masses(m_tot, imf_type=imf, m_min=m_min,
                                   m_max=m_max, force_resample=True)
                ms = mock.masses
                assert np.min(ms) > m_min.to(apu.Msun).value,\
                    'Masses sampled below m_min when astropy used'
                assert np.max(ms) < m_max.to(apu.Msun).value,\
                    'Masses sampled above m_max when astropy used'
                m_tot_diff = np.fabs(np.sum(ms)-m_tot.to(apu.Msun).value)
                assert m_tot_diff < m_max.to(apu.Msun).value,\
                    'Total mass sampled does not match m_tot when astropy used'
                
        
def test_sample_mass_distribution_matches_IMF():
    '''test_sample_mass_distribution_matches_IMF:
    
    Test that sample masses are distributed correctly matching the IMF
    '''
    m_min = 0.1
    m_max = 0.9
    m_tot = 1e5
    denspot = potential.HernquistPotential()
    mock = APOGEEMock(denspot=denspot)

    imf_lims = np.array([0.3,0.4,0.5,0.6,0.7,0.8])
    imf_types = ['chabrier','kroupa']
    imf_fns = [chabrier_imf,kroupa_imf]
    for imf,imf_fn in zip(imf_types,imf_fns):
        mock.sample_masses(m_tot, imf_type=imf, m_min=m_min, 
                           m_max=m_max, force_resample=True)
        ms = mock.masses

        m_lim_int_callable = lambda x: x*imf_fn(x)
        imf_mass_tot = scipy.integrate.quad(m_lim_int_callable, a=m_min, 
                                            b=m_max)[0]

        for lim in imf_lims:
            imf_lim_mass = scipy.integrate.quad(m_lim_int_callable, a=m_min, 
                                                b=lim)[0]
            imf_lim_mass_frac = imf_lim_mass/imf_mass_tot
            sample_lim_mass_frac = np.sum(ms[ms<lim])/m_tot
            assert np.fabs(sample_lim_mass_frac-imf_lim_mass_frac)/imf_lim_mass_frac\
                < 0.01, 'Sample mass fraction does not match IMF integration'

def test_sample_radial_mass_profile():
    '''test_sample_radial_mass_profile:
    
    Test that radial mass profile of the position samples matches the input 
    density profile
    '''
    ro,vo = 8.,220.
    denspots = [potential.HernquistPotential(),
                potential.PowerSphericalPotential(alpha=2.),
                potential.PowerSphericalPotential(alpha=3.1),
                potential.PowerSphericalPotentialwCutoff(alpha=2.5,rc=3.),
                potential.PowerSphericalPotentialwCutoff(alpha=3.5,rc=5.),
                potential.NFWPotential()
               ]

    rmin = 2./ro
    rmax = 70./ro
    rlims = np.arange(10.,70.,10.)/ro

    for denspot in denspots:
        print(denspot)
        mock = APOGEEMock(denspot=denspot)
        _n_fake_masses = int(1e6)
        _fake_masses = np.ones(_n_fake_masses) # Hack
        mock.masses = _fake_masses
        mock.sample_positions(r_min=2./ro,r_max=70./ro)
        rs = mock.orbs.r(use_physical=False)
        pot_mtot = potential.mass(denspot,rmax,use_physical=False)-\
            potential.mass(denspot,rmin,use_physical=False)
        for rlim in rlims:
            pot_mfrac = (potential.mass(denspot,rlim,use_physical=False)-\
                potential.mass(denspot,rmin,use_physical=False))/pot_mtot
            sample_mfrac = len(np.where(rs < rlim)[0])/_n_fake_masses
            print(np.fabs(pot_mfrac-sample_mfrac)/pot_mfrac)
            assert np.fabs(pot_mfrac-sample_mfrac)/pot_mfrac < 5e-3,\
                'sample mass profile does not match denspot'

def test_lbIndx_matches_mwdust():
    '''test_lbIndx_matches_mwdust:
    
    Test that calculated lbIndx values match those from mwdust
    '''
    dmap = mwdust.Combined19(filter='2MASS H')
    # Mock and orbits
    mock = APOGEEMock(
        denspot=potential.HernquistPotential(ro=_ro,vo=_vo))
    n = int(1e2)
    # Sample random galactic coordintes, distances out to 50 kpc
    ll=np.random.random(n)*360.
    bb=np.arccos(2*np.random.random(n)-1)*180./np.pi-90.
    dist = np.random.random(size=n)*50.
    dm = 5.*np.log10(dist)+10.
    mul,mub,vlos = np.zeros_like(ll), np.zeros_like(ll), np.zeros_like(ll)
    vxvvs = np.array([ll,bb,dist,mul,mub,vlos]).T
    orbs = orbit.Orbit(vxvvs,lb=True)
    # Calculate lbIndx using mwdust and mock, then compare
    lbIndx_dmap = np.zeros(n)
    for i in range(n):
        lbIndx_dmap[i] = dmap._lbIndx(ll[i],bb[i])
    lbIndx_mock = mock._get_lbIndx(orbs,dmap)
    assert np.all(lbIndx_dmap == lbIndx_mock.astype(int)),\
        'lbIndx values from mock do not match values from mwdust'

def test_extinction_matches_mwdust():
    '''test_extinction_matches_mwdust:
    
    Test that calculated H-band extinctions match those from mwdust
    '''
    dmaps = [mwdust.Combined19,mwdust.Combined15]
    tol = 1e-10 # Should be exact match
    # Mock and orbits
    mock = APOGEEMock(
        denspot=potential.HernquistPotential(ro=_ro,vo=_vo))
    n = int(1e2)
    # Sample random galactic coordintes, distances out to 50 kpc
    ll=np.random.random(n)*360.
    bb=np.arccos(2*np.random.random(n)-1)*180./np.pi-90.
    dist = np.random.random(size=n)*50.
    dm = 5.*np.log10(dist)+10.
    mul,mub,vlos = np.zeros_like(ll), np.zeros_like(ll), np.zeros_like(ll)
    vxvvs = np.array([ll,bb,dist,mul,mub,vlos]).T
    orbs = orbit.Orbit(vxvvs,lb=True)
    for i in range(len(dmaps)):
        dmap = dmaps[i](filter='2MASS H')
        # Calculate AH using mwdust and mock, then compare
        lbIndx_mock = mock._get_lbIndx(orbs,dmap)
        AH_mock = mock._calculate_AH(dmap,lbIndx_mock,5.*np.log10(dist)+10.)
        AH_dmap = np.zeros(n)
        for j in range(n):
            #import pdb
            #pdb.set_trace()
            AH_dmap[j] = dmap(ll[j],bb[j],dist[j])
        assert np.all(np.fabs(AH_mock-AH_dmap)<tol),\
            'AH values from mock do not match values from mwdust for dust map'\
            +str(dmap)