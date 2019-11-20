import numpy as np
from scipy.optimize import minimize

def chi2_pol_angle(alpha, obs_map_frame, coadd_map_frame, map_covariance='1'):
    '''
    Chi square used to fit a polarization angle per observation.

    Parameters
    ----------
    alpha : float
        Polarization angle
    obs_map_frame : G3Frame
        Map frame for one observation
    coadd_map_frame : G3Frame
        Map frame for the coadded observations
    map_covariance : string
        String to select covariance matrix option:
            '1': Identity matrix
        Note that there are too many pixels in the map to pass a true
        covariance matrix, so we need to be clever about how we handle this, 
        hence the options keyed by strings
    
    Returns
    -------
    chi2 : float
        The value of chi-square evaluated at the given polarization angle
    '''
    Q = obs_map_frame['Q']
    U = obs_map_frame['U']
    Qcoadd = coadd_map_frame['Q']
    Ucoadd = coadd_map_frame['U']

    if map_covariance == '1':
        Qresidual = Q - (1/2)*np.cos(2*alpha)*Qcoadd + (1/2)*np.sin(2*alpha)*Ucoadd
        Uresidual = U - (1/2)*np.cos(2*alpha)*Ucoadd - (1/2)*np.sin(2*alpha)*Qcoadd

        chi2 = np.sum(Qresidual * Qresidual) + np.sum(Uresidual * Uresidual)

    return chi2


def fit_pol_angle(obs_map_frame, coadd_map_frame, map_covariance='1'):
    '''
    Fit for the polarization angle for a single observation by minimizing
    the map-space chi-square.

    Parameters
    ----------
    obs_map_frame : G3Frame
        Map frame for one observation
    coadd_map_frame : G3Frame
        Map frame for the coadded observations
    map_covariance : string
        String to select covariance matrix option:
            '1': Identity matrix
        Note that there are too many pixels in the map to pass a true
        covariance matrix, so we need to be clever about how we handle this, 
        hence the options keyed by strings
    
    Returns
    -------
    chi2 : float
        The value of chi-square evaluated at the given polarization angle
    '''
    result = minimize(chi2_pol_angle, 0, args=(obs_map_frame, coadd_map_frame,
                                               map_covariance), method='Powell')
    return result
