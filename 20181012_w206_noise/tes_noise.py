import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const
import scipy.integrate as sciint


def shot_noise(nu, power):
    return np.sqrt(2. * const.Planck * nu * power)

def correlation_noise(nu, power, delta_nu, correlation):
    return np.sqrt(2 * correlation * power**2. / delta_nu)

def tes_phonon_noise_P(Tbolo, G, gamma):
    return np.sqrt(4. * gamma * const.Boltzmann * G * Tbolo**2.)

# TES Johnson noise should be a small contribution, but seems to end up too large
# at high frequencies where not suppressed by ETF, for our standard TES parameters...
# is this correct?
def tes_johnson_noise_P(nu, T, Rfrac, k, Tc, Rn, R_L, C, Popt=0):
    R_0 = Rn*Rfrac
    I_0 = np.sqrt(Psat(k, Tc, T, Popt) / R_0)
    tau = C / G(k, Tc)
    return np.sqrt(4. * const.Boltzmann * Tc * I_0**2 * R_0) * \
            np.sqrt(1 + (2*np.pi * nu)**2. * tau**2.)

def load_johnson_noise_I(nu, T_L, R_L, R_bolo, L):
    S = np.sqrt(4. * const.Boltzmann * R_L * T_L) / (R_L + R_bolo + 1j * (2.*np.pi) * nu * L)
    return np.abs(S)

def dIdP(nu, R_L, R_0, I_0, L, alpha, beta, C, G, Tc):
    P_J = I_0**2. * R_0
    loopgain = P_J * alpha / (G * Tc)
    tau = C / G
    tau_el = L / (R_L + R_0*(1 + beta))
    tau_I = tau / (1 - loopgain)

    S = (-1. / (I_0 * R_0)) * ( L / (tau_el * R_0 * loopgain) +
                                (1 - R_L / R_0) +
                                1j * 2.*np.pi*nu * (L*tau / (R_0*loopgain)) * (1./tau_I + 1./tau_el) -
                                (2.*np.pi*nu)**2. * tau * L / (loopgain * R_0))**-1
    return np.abs(S)


def dIdP_2(nu, T, Rfrac, k, Tc, Rn, R_L, L, alpha, beta, C, Popt=0):
    G = 3.*k*(Tc**2.)
    R_0 = Rn*Rfrac
    I_0 = np.sqrt(Psat(k, Tc, T, Popt) / R_0)
    P_J = I_0**2. * R_0
    loopgain = P_J * alpha / (G * Tc)
    G = 3.*k*(Tc**2.)
    tau = C / G
    tau_el = L / (R_L + R_0*(1 + beta))
    tau_I = tau / (1 - loopgain)

    S = (-np.sqrt(2.) / (I_0 * R_0)) * ( L / (tau_el * R_0 * loopgain) +
                                (1 - R_L / R_0) +
                                1j * 2.*np.pi*nu * (L*tau / (R_0*loopgain)) * (1./tau_I + 1./tau_el) -
                                (2.*np.pi*nu)**2. * tau * L / (loopgain * R_0))**-1
    return np.abs(S)

def Vbias(Psat, Popt, Rbolo):
    PJ = Psat - Popt
    Vbias = np.sqrt(PJ * Rbolo)
    return Vbias

def Psat(k, Tc, T, Popt=0):
    return k*(Tc**3 - T**3) - Popt

def G(k, Tc):
    return 3.*k*(Tc**2.)

def readout_noise_I(Ibase, Lsquid, fbias, Rbolo):
    return Ibase * np.abs(1 + 1j*2*np.pi*fbias*Lsquid / Rbolo)

def tau_natural(k, Tc, C):
    G = 3.*k*(Tc**2.)
    tau = C / G

def planck_spectral_density(nu, temp):
    dPdnu = hPlanck * nu / (np.exp(hPlanck * nu / (kB * temp)) - 1) * 1e12
    return dPdnu

def planck_power(temp, nu_min, nu_max):
    power = sciint.quad(planck_spectral_density, a=nu_min, b=nu_max, args=(temp))
    return power
