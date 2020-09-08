import numpy as np
import camb
from camb import model
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.constants import c
from scipy.special import erf


# Set up CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(10000, lens_potential_accuracy=0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)


# Set up CMB TT spectrum
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
ls = np.arange(totCL.shape[0])
DlTT = interp1d(ls, totCL[:,0])
ClTT = interp1d(ls, totCL[:,0] / (ls*(ls+1)) * (2*np.pi))


# Set up P_eta^perp(k) from Simone
Peta_data = np.loadtxt('Peta_data.csv', delimiter=',')
Peta_data = np.vstack([[0,Peta_data[0,1]], Peta_data, [10,0], [100,0]])
Peta_interp = interp1d(Peta_data[:,0], Peta_data[:,1])


# Late-time kSZ
# This is stolen directly from Shaw, et al. (1109.0553).
# First: source redshift dependence of late-time kSZ ("CSF" curve)
data = np.loadtxt('dDldz_ksz_csf_data.csv', delimiter=',')
data = np.vstack([[0,0], data, [100, data[-1,1]]])
dDlkSZdz_Shaw_interp = interp1d(data[:,0], data[:,1])
def dDlkSZdz_Shaw(z):
    return dDlkSZdz_Shaw_interp(z) * 0.5 * (1 - erf((z - 8.8) / 1.0))

# Second: angular spectrum
data = np.sort(np.loadtxt('Dl_ksz.csv', delimiter=','), axis=0)
data = np.vstack([[0,0], data, [10000, data[-1,1]]])
DlkSZ_Shaw = interp1d(data[:,0], data[:,1])

# Third: put the angular- and redshift-dependence together
def dCkSZdz_late_shaw(ell, z, A_late=1, alpha_late=0):
    '''
    Calculates (d(C_ell^kSZ) / dz) |_late

    Parameters
    ----------
    ell : float
        Multipole
    z : float
        Redshift
    A_late : float
        Amplitude scaling factor for kSZ model (formalism described in 
        equation 12 of 1803.07036)
    alpha_late : float
        Spectral index scaling factor for kSZ model (described in same place 
        as A_late)

    Returns
    -------
    dCkSZdz : float
        Result
    '''
    dDkSZdz = DlkSZ_Shaw(ell) * (dDlkSZdz_Shaw(z) / DlkSZ_Shaw(3000))
    dCkSZdz = dDkSZdz / (ell*(ell+1)) * (2*np.pi) * A_late * (ell**alpha_late)
    return dCkSZdz


def Dl_3000_reion_ksz_uK2(z_re=8.8, delta_z_re=1.2):
    '''
    Amplitude at ell=3000 of the patchy kSZ power spectrum in D_ell. Amplitude
    is from equation 6 of 2002.06197.

    Parameters
    ----------
    z_re : float
        Mean redshift of reionization
    delta_z_re : float
        Duration of reionization. This is assumed to be the FWHM amplitude of
        a gaussian describing the kSZ source redshift of reionization.

    Returns
    -------
    D_ell : float
        D_ell at ell=3000
    '''
    return 2.03 * (((1+z_re) / 11) - 0.12) * (delta_z_re / 1.05)**0.51


# Reionization kSZ
def dCkSZdz_reion(ell, z, z_re=8.8, delta_z_re=1.2):
    '''
    Patchy reionization contribution to the kSZ effect. Spectrum in ell is assumed to be flat in D_ell.

    Parameters
    ----------
    ell : float
        Multipole
    z : float
        Redshift
    z_re : float
        Mean redshift of reionization
    delta_z_re : float
        Duration of reionization. This is assumed to be the FWHM amplitude of
        a gaussian describing the kSZ source redshift of reionization.

    Returns
    -------
    '''
    Dl_3000_ksz_uK2 = Dl_3000_reion_ksz_uK2(z_re, delta_z_re)
    Cl_kSZ = (Dl_3000_ksz_uK2 / (ell*(ell+1)) * 2 * np.pi)

    sigma_z_re = delta_z_re / np.sqrt(2.*np.log(2.))
    norm_pdf = 1./np.sqrt(2*np.pi*(sigma_z_re**2)) * \
               np.exp(-1*((z - z_re) / sigma_z_re)**2 / 2)
    return norm_pdf * Cl_kSZ


# kSZ filter function
# See text of 1607.01769 between equations 4 and 5.
def W_s(ell, Dl_3000_ksz_uK2=3):
    return np.sqrt(Dl_3000_ksz_uK2 * 2*np.pi / (ell*(ell+1))) / \
          (ClTT(ell) + Dl_3000_ksz_uK2 * 2*np.pi / (ell*(ell+1)))


# dK/dz
def dKdz(z, components=['reion', 'late'], z_re=8.8, delta_z_re=1.2, A_late=1, alpha_late=0):
    '''
    Calculate dK/dz, the contribution to the K statistic as a function of 
    redshift. This is defined via equation 7 of 1607.01769.

    Parameters
    ----------
    z : float
        Redshift
    component : list of str
        The components to include in the kSZ power. Options include 'reion'
        and 'late'.

    Returns
    -------
    dKdz : float
        dK/dz at redshift z
    '''
    def integrand(ell):
        dCkSZdz = 0
        if 'reion' in components:
            dCkSZdz += dCkSZdz_reion(ell, z, z_re=8.8, delta_z_re=1.2)
        if 'late' in components:
            dCkSZdz += dCkSZdz_late_shaw(ell, z, A_late=1, alpha_late=0)
        return ell / (2*np.pi) * W_s(ell)**2 * dCkSZdz

    result, _ = quad(integrand, 100, 10000)
    return result

# Interpolated version of dK/dz for faster evaluation.
# NB: Evaluating dK/dz is slow due to numerical integration, which slows down
# importing the module.
z_interp = np.linspace(0, 15, 100)
dKdz_interp = {'reion': interp1d(z_interp, [dKdz(z, components=['reion']) for z in z_interp]),
               'late': interp1d(z_interp, [dKdz(z, components=['late']) for z in z_interp])}


# dC_L^KK / dz
def dCLKKdz(z, L, components=['reion', 'late'], interp=dKdz_interp, z_re=8.8, delta_z_re=1.2, A_late=1, alpha_late=0):
    '''
    dC_L^KK / dz: contribution to C_L^KK per unit redshift; from equation 
    9 of 1607.01769.

    Parameters
    ----------
    z : float
        Redshift
    L : float
        Multipole
    component : list of str
        The components to include in the kSZ power. Options include 'reion'
        and 'late'.
    interp : dict of scipy.interpolate.interpolate.interp1d object
        Use an interpolation function for dK/dz for faster evaluation.
        Argument is assumed to be a dictionary with values containing
        interpolating functions for different components of the kSZ signal.
        For example:
            dKdz_interp = {'reion': interp1d(z_interp, [dKdz(z, components=['reion']) for z in z_interp]),
                           'late': interp1d(z_interp, [dKdz(z, components=['late']) for z in z_interp])}
        Set to `None` to not use interpolation.

    Returns
    -------
    dCLKKdz : float
        dC_L^KK / dz
    '''
    if interp is not None:
        dKdz_eval = 0
        for component in components:
            dKdz_eval += interp[component](z)
    else:
        dKdz_eval = dKdz(z, component, z_re, delta_z_re, A_late=1, alpha_late=0)
        
    # NB: The factor of c/1000 is introduced to cancel the km/s in the Hubble
    # parameter. It is divided by 1000 because `c` from scipy.constants is in
    # m/s by default.
    out = results.hubble_parameter(z) / (c/1000) / \
         (results.comoving_radial_distance(z)**2) * \
         (dKdz_eval**2) * Peta_interp(L / results.comoving_radial_distance(z))
    return out


# C_L^KK
def CLKK(L, components=['reion', 'late'], interp=dKdz_interp):
    '''
    C_L^KK from equation 9 of 1607.01769.

    Parameters
    ----------
    L : float
        Multipole
    component : list of str
        The components to include in the kSZ power. Options include 'reion'
        and 'late'.
    interp : dict of scipy.interpolate.interpolate.interp1d object
        Use an interpolation function for dK/dz for faster evaluation.
        Argument is assumed to be a dictionary with values containing
        interpolating functions for different components of the kSZ signal.
        For example:
            dKdz_interp = {'reion': interp1d(z_interp, [dKdz(z, components=['reion']) for z in z_interp]),
                           'late': interp1d(z_interp, [dKdz(z, components=['late']) for z in z_interp])}
        Set to `None` to not use interpolation.

    Returns
    -------
    dCLKKdz : float
        dC_L^KK / dz
    '''
    def integrand(z):
        return dCLKKdz(z, L, components, interp)
    result, _ = quad(integrand, 0.01, 13)
    return result


# Ktotal
def Ktotal(Dl_3000_ksz_uK2=3):
    def integrand(ell):
        return ell / (2*np.pi) * W_s(ell)**2 * (ClTT(ell) + Dl_3000_ksz_uK2 / (ell*(ell+1)) * 2*np.pi)
    result, _ = quad(integrand, 100, 10000)
    return result


# Noise
def NLKK(L, noise_uKarcmin=2):
    '''
    4-point estimator reconstruction noise

    Parameters
    ----------
    L : float
        Multipole
    noise_uKarcmin : float
        Map noise in uK-arcmin

    Returns
    -------
    NLKK : float
        Noise level of 
    '''
    noise_Cl_uK2 = (noise_uKarcmin * np.pi / 10800)**2
    def integrand(ell):
        return ell * W_s(ell)**2 * W_s(np.abs(L - ell))**2 * \
               (ClTT(ell) + noise_Cl_uK2) * \
               (ClTT(np.abs(L - ell)) + noise_Cl_uK2)
    result, _ = quad(integrand, 100, 10000)
    return result