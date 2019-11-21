import numpy as np
from scipy.optimize import minimize
from spt3g.mapmaker import remove_weight

def chi2_pol_angle(alpha, obs_map_frame, coadd_map_frame, map_covariance='1',
                   std_q=None, std_u=None):
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
            'std': Identity matrix scaled by the std of the pixels in the map
               with finite value
        Note that there are too many pixels in the map to pass a true
        covariance matrix, so we need to be clever about how we handle this, 
        hence the options keyed by strings
    std_q : float
        The factor by which the Q part of an identity covariance matrix is
        scaled
    std_u : float
        Same as std_q, but applied to the U part of the covariance

    Returns
    -------
    chi2 : float
        The value of chi-square evaluated at the given polarization angle
    '''
    # need to remove weights
    _, Q, U = remove_weight(obs_map_frame['T'], obs_map_frame['Q'],
                            obs_map_frame['U'], obs_map_frame['Wpol'])
    _, Qcoadd, Ucoadd = remove_weight(coadd_map_frame['T'], coadd_map_frame['Q'],
                                      coadd_map_frame['U'], coadd_map_frame['Wpol'])

    # different heuristic choices of covariance matrix
    if map_covariance == '1':
        Qresidual = Q - (1/2)*np.cos(2*alpha)*Qcoadd + (1/2)*np.sin(2*alpha)*Ucoadd
        Uresidual = U - (1/2)*np.cos(2*alpha)*Ucoadd - (1/2)*np.sin(2*alpha)*Qcoadd
        Qresidual_arr = np.array(Qresidual)
        Qresidual_arr = Qresidual_arr[np.isfinite(Qresidual_arr)]
        Uresidual_arr = np.array(Uresidual)
        Uresidual_arr = Uresidual_arr[np.isfinite(Uresidual_arr)]

        chi2 = np.sum(Qresidual_arr * Qresidual_arr) + np.sum(Uresidual_arr * Uresidual_arr)

    if map_covariance == 'std':
        if std_q is None or std_u is None:
            raise ValueError('Argument `map_covariance` set to `std`, but '
                             '`std_q` and/or `std_u` not specified.')
        Qresidual = Q - (1/2)*np.cos(2*alpha)*Qcoadd + (1/2)*np.sin(2*alpha)*Ucoadd
        Uresidual = U - (1/2)*np.cos(2*alpha)*Ucoadd - (1/2)*np.sin(2*alpha)*Qcoadd
        Qresidual_arr = np.array(Qresidual)
        Qresidual_arr = Qresidual_arr[np.isfinite(Qresidual_arr)]
        Uresidual_arr = np.array(Uresidual)
        Uresidual_arr = Uresidual_arr[np.isfinite(Uresidual_arr)]

        chi2 = np.sum(Qresidual_arr * Qresidual_arr) / (std_q**2) + \
               np.sum(Uresidual_arr * Uresidual_arr) / (std_u**2)

    return chi2


def fit_pol_angle(obs_map_frame, coadd_map_frame, map_covariance='1',
                  std_q=None, std_u=None):
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
                                               map_covariance, std_q, std_u),
                      method='Powell')
    return result
