# ----------------------------------------------------------------------------
#
# TITLE - util.py
# AUTHOR - James Lane
# CONTENTS - IMF functions, Xi <-> r conversion, metallicity conversions
#
# ----------------------------------------------------------------------------

import numpy as np
import scipy.integrate
from galpy import orbit

# def chabrier01_lognormal():
#     '''chabrier01_lognormal:
#     '''

# def chabrier01_exponential():
#     '''chabrier01_exponential:
#     '''
    
# def chabrier03_lognormal():
#     '''chabrier01_lognormal:
#     '''

# def chabrier05_lognormal():
#     '''chabrier01_lognormal:
#     '''
    

# def _chabrier_lognormal(m,m0=0.,sigma=1.):
#     '''_chabrier_lognormal:
    
#     Lognormal-type Chabrier initial mass function
    
#     Args:
    
#     Returns:
#         dn/dm (np.ndarray) - Value of the IMF for given linear mass interval
#     '''
#     dNdlogm = np.exp(-(np.log10(m)-np.log10(m0))/(2.*sigma**2.)
#     dlogmdm = 1./m/np.log(10.)
#     return A*dNdlogm*dlogmdm

def chabrier_imf(m,k=0.0193,A=1.):
    '''chabrier_imf:
    
    Chabrier initial mass function
    
    Args:
        m (np.ndarray) - Masses [solar]
        k (float) - scale factor to apply to the IMF where m>1 to equalize it 
            to the IMF where m<1
        A (float) - arbitrary scale factor
    
    Returns:
        Nm (np.ndarray) - Value of the IMF for given masses
    '''
    k = 0.0193 # Equalizes m<1 and m>1 at m=1
    a = 2.3
    
    if not isinstance(m,np.ndarray):
        m = np.atleast_1d(m)
    ##fi
    
    where_m_gt_1 = m>1
    Nm = np.empty(len(m))
    Nm[~where_m_gt_1] = (0.158/(np.log(10)*m[~where_m_gt_1]))\
                        *np.exp(-(np.log10(m[~where_m_gt_1])-np.log10(0.08))**2\
                               /(2*0.69**2))
    Nm[where_m_gt_1] = k*m[where_m_gt_1]**(-a)
    Nm[m<0.01] = 0
    return A*Nm
#def

def kroupa_imf(m,k1=1.):
    '''kroupa_imf:
    
    Kroupa initial mass function
    
    Args:
        m (np.ndarray) - Masses [solar]
        k1 (float) - Normalization for the first power law (all other follow
            to make sure boundaries are continuous)
    
    Returns:
        Nm (np.ndarray) - Value of the IMF for given masses  
    '''
    a1,a2,a3 = 0.3,1.3,2.3
    k2 = 0.08*k1
    k3 = 0.5*k2
    
    if not isinstance(m,np.ndarray):
        m = np.atleast_1d(m)
    ##fi
    
    where_m_1 = np.logical_and(m>=0.01,m<0.08)
    where_m_2 = np.logical_and(m>=0.08,m<0.5)
    where_m_3 = m>=0.5
    Nm = np.empty(len(m))
    Nm[where_m_1] = k1*m[where_m_1]**(-a1)
    Nm[where_m_2] = k2*m[where_m_2]**(-a2)
    Nm[where_m_3] = k3*m[where_m_3]**(-a3)
    Nm[m<0.01] = 0
    return Nm
#def

def _cimf(imf,a,b,intargs=()):
    '''_cimf:

    Calculate the cumulative of the initial mass function

    Args:
        imf (callable) - Initial mass function
        a (float) - minimum mass integration bound
        b (float) - maximum mass integration bound
        intargs (dict, optional) - dictionary of args for imf

    Returns:
        cimf (float) - Integral of the initial mass function
    '''
    return scipy.integrate.quad(imf,a,b,args=intargs)[0]
#def

def _r_to_xi(r,a=1.):
    '''_r_to_xi:

    Convert r to the variable xi
    '''
    out= np.divide((r/a-1.),(r/a+1.),where=True^np.isinf(r))
    if np.any(np.isinf(r)):
        if hasattr(r,'__len__'):
            out[np.isinf(r)]= 1.
        else:
            return 1.
    return out
#def

def _xi_to_r(xi,a=1.):
    '''_xi_to_r:

    Convert the variable xi to r
    '''
    return a*np.divide(1.+xi,1.-xi)
#def

def Z2FEH(z,zsolar=None):
    '''Z2FEH:
    
    Convert Z to FeH assuming zsolar
    
    Args:
        z (np.array) - metal fraction to convert to [Fe/H]
        zsolar (float, optional) - solar metallicity
    
    Returns:
        feh (np.array) - [Fe/H]
    '''
    # if parsec:
    #     if zsolar is None: zsolar= 0.0152
    #     return np.log10(z/(1.-0.2485-2.78*z))-math.log10(zsolar/(1.-0.2485-2.78*zsolar))
    # else:
    if zsolar is None: zsolar= 0.0196
    return np.log10(z)-np.log10(zsolar)

def FEH2Z(feh,zsolar=None):
    '''FEH2Z:
    
    Convert FeH to Z assuming zsolar
    
    Args:
        feh (np.array) - [Fe/H] to convert to metal fraction
        zsolar (float, optional) - solar metallicity
    
    Returns:
        z (np.array) - metallicity
    '''
    # if parsec:
    #     if zsolar is None: zsolar= 0.0152
    #     zx= 10.**(feh+math.log10(zsolar/(1.-0.2485-2.78*zsolar)))
    #     return (zx-0.2485*zx)/(2.78*zx+1.)
    # else:
    if zsolar is None: zsolar= 0.0196
    return 10.**(feh+np.log10(zsolar))

def join_orbs(orbs):
    '''join_orbs:
    
    Join a list of orbit.Orbit objects together. They must share ro,vo
    
    Args:
        orbs (orbit.Orbit) - list of individual orbit.Orbit objects
    
    Returns
        orbs_joined (orbit.Orbit) - Joined orbit.Orbit object
    '''
    for i,o in enumerate(orbs):
        if i == 0:
            ro = o._ro
            vo = o._vo
            vxvvs = o._call_internal()
        else:
            assert ro==o._ro and vo==o._vo, 'ro and/or vo do not match'
            vxvvs = np.append(vxvvs, o._call_internal(), axis=1)
    return orbit.Orbit(vxvvs.T,ro=ro,vo=vo)