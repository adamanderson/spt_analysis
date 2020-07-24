from spt3g import core, maps
import numpy as np
import matplotlib.pyplot as plt


def calculate_rho(data_fr, coadd_fr, freq_factor, return_err=False, use_t=False):
    '''
    Takes (un)weighted T/Q/U maps and an unweighted T/Q/U coadd,
    as well as the weights from the map you want rho for,
    and calculates the polarization rotation angle rho.
    '''
    maps.RemoveWeights(data_fr)
    maps.RemoveWeights(coadd_fr)

    q = data_fr['Q']
    u = data_fr['U']
    qc = coadd_fr['Q']
    uc = coadd_fr['U']
    weights = data_fr['Wpol']

    qweight = weights.QQ
    uweight = weights.UU
    quweight = weights.QU
    if use_t:
        t = data_fr['T']
        tc = coadd_fr['T']
        tweight = weights.TT
        tqweight = weights.TQ
        tuweight = weights.TU

        num = np.asarray(tuweight*qc*(t-tc) - tqweight*uc*(t-tc) - qweight*uc*(q-qc) + \
                         uweight*qc*(u-uc) + quweight*(qc*q - qc**2 + uc**2 - uc*u))
    else:
        num = np.asarray(qweight*uc*(qc-q) + uweight*qc*(u-uc) + \
                         quweight*(qc*q - qc**2 + uc**2 - uc*u))
    den = np.asarray(qweight*uc**2 + uweight*qc**2 - 2*quweight*qc*uc)

    #num[num == np.inf] = 0
    #den[den == np.inf] = 0
    rho = np.nansum(num) / np.nansum(den)

    if return_err:
        try:
            err = get_errs(rho, data_fr, coadd_fr, freq_factor, use_t=use_t)
        except:
            err = np.nan
        return rho, err
    else:
        return rho


def get_errs(min_rho, data_fr, coadd_fr, freq_factor, use_t=False):
    maps.RemoveWeights(data_fr)
    maps.RemoveWeights(coadd_fr)

    chi2s = []
    rhos = np.linspace(min_rho-5.*(np.pi/180), min_rho+5.*(np.pi/180), 11)

    for rho in rhos:
        chi2s.append(calculate_chi2(data_fr, coadd_fr, freq_factor, rho, use_t=use_t))
    p = np.polyfit(rhos, np.array(chi2s), deg=2)

    p[2] = p[2] - chi2s[5] - 1
    err_1, err_2 = np.roots(p)
    err_down = np.min([err_1, err_2])
    err_up = np.max([err_1, err_2])

    return np.mean((min_rho - err_down, err_up - min_rho))


def calc_chi2_adam(t, tc, q, qc, u, uc, wt, delta_f, pol_rotation, return_map=True):
    if type(pol_rotation) is not np.ndarray:
        pol_rotation = np.array([pol_rotation])
        
    chi2_map = np.zeros(len(pol_rotation))
        
    for jpol, pol_rot in enumerate(pol_rotation):
        npixels = len(q)
        #delta_f = 3
        chi2_per_pixel = np.zeros((200,200))
        for ipixel in np.arange(800,1000):
            for jpixel in np.arange(1000, 1200):
                ipixel = int(ipixel)
                jpixel = int(jpixel)
                if np.isfinite(mapdata_noweight['T'][ipixel, jpixel]):
                    weights = mapdata['Wpol'][ipixel, jpixel]
                    tqu = np.array([mapdata_noweight['T'][ipixel, jpixel] - coadd_noweight['T'][ipixel, jpixel],
                                    mapdata_noweight['Q'][ipixel, jpixel] - coadd_noweight['Q'][ipixel, jpixel] + \
                                        pol_rot*coadd_noweight['U'][ipixel, jpixel],
                                    mapdata_noweight['U'][ipixel, jpixel] - coadd_noweight['U'][ipixel, jpixel] - \
                                        pol_rot*coadd_noweight['Q'][ipixel, jpixel]])
                    chi2_per_pixel[ipixel-800, jpixel-1000] = np.matmul(np.matmul(tqu.transpose(), weights/(delta_f**2)), tqu)
                    chi2_map[jpol] += chi2_per_pixel[ipixel-800, jpixel-1000]
        #     chi2_per_pixel = np.array(chi2_per_pixel)
        print(chi2_map[jpol])
    if calc_per_pixel:
        return chi2_map, chi2_per_pixel
    else:
        return chi2_map


def calculate_chi2(data_fr, coadd_fr, freq_factor, rho, use_t=False, return_map=False):
    '''
    Takes (un)weighted T/Q/U maps and an (un)weighted T/Q/U coadd,
    and calculates the chi^2 for a given input rotation
    angle rho.
    '''
    maps.RemoveWeights(data_fr)
    maps.RemoveWeights(coadd_fr)

    q = data_fr['Q']
    u = data_fr['U']
    qc = coadd_fr['Q']
    uc = coadd_fr['U']
    weights = data_fr['Wpol']

    qweight = weights.QQ
    uweight = weights.UU
    quweight = weights.QU

    chiqq = qweight * (q - qc + rho*uc)**2
    chiuu = uweight * (u - uc - rho*qc)**2
    chiqu = 2 * quweight * (q - qc + rho*uc) * (u - uc - rho*qc)

    if use_t:
        t = data_fr['T']
        tc = coadd_fr['T']
        tweight = weights.TT
        tqweight = weights.TQ
        tuweight = weights.TU
        chitt = tweight * (t - tc)**2
        chitq = 2 * tqweight * ((t-tc)*(q-qc) + rho*uc*(t-tc))
        chitu = 2 * tuweight * ((t-tc)*(u-uc) - rho*qc*(t-tc))

        chi2_map = (chiqq + chiuu + chiqu + chitt + chitq + chitu) / (freq_factor**2)
    else:
        chi2_map = (chiqq + chiuu + chiqu) / (freq_factor**2)

    chi2 = np.nansum(chi2_map)

    if return_map:
        return chi2, chi2_map
    else:
        return chi2


def generate_signal(amp, freq, times, phase=None):
    '''
    amp must be in same units as rotation angles.
    '''
    if phase == None:
        phase = np.random.uniform(low=0, high=2*np.pi)

    #freq = mass * ((60*60*24) / (2*np.pi*6.528e-16))
    #conversion factor from Adam's ipy notebook
    #m_axion_2019 = freqs * ((2*np.pi*6.528e-16)/(60*60*24))

    return amp*np.sin(2*np.pi*freq*times + phase)


def inject_signal_in_map(amp, data_fr, coadd_fr):
    '''
    Inject a map-space signal, rotating Q into U and vice-versa.
    amp must be in radians.
    '''
    maps.RemoveWeights(data_fr)
    maps.RemoveWeights(coadd_fr)

    q = data_fr.pop('Q')
    u = data_fr.pop('U')
    qc = coadd_fr['Q']
    uc = coadd_fr['U']
    data_fr['Q'] = q - amp*uc
    data_fr['U'] = u + amp*qc
    return data_fr


def plot_parameter_space_estimate(median_upper_lims_yr, m_axion_yr, band, t_used=False):
    m_axion = np.logspace(-23, -18, 100)

    fig, ax1 = plt.subplots()

    # CAST solar axion limit
    g_limit_cast = 6.6e-11 # from 1705.02290
    ax1.loglog(m_axion, g_limit_cast*np.ones(m_axion.shape))
    ax1.fill_between(m_axion, g_limit_cast*np.ones(m_axion.shape), 1, alpha=0.2)
    ax1.text(3.5e-19, 8e-11, s='CAST', color='C0', alpha=0.6)

    # Planck "washout" limit
    g_limit_planck = 9.6e-13 * (m_axion / 1e-21) # from equation (73) of 1903.02666
    ax1.loglog(m_axion, g_limit_planck, color='C2')
    ax1.fill_between(m_axion, g_limit_planck, 1, alpha=0.2, color='C2')
    ax1.text(1.3e-23, 2e-13, s='Planck "washout"',
             rotation=33, color='C2', alpha=0.6)

    # Cosmic variance limit
    g_limit_cv = 3.6e-13 * (m_axion / 1e-21) # from equation (79) of 1903.02666
    ax1.loglog(m_axion, g_limit_cv, '--', color='C2', alpha=0.5)
    #plt.fill_between(m_axion, g_limit_cv, 1, alpha=0.2)
    ax1.text(2.7e-23, 1.3e-13, s='Cosmic variance',
                 rotation=33, color='C2', alpha=0.6)

    # small-scale structure (limit is very approximate)
    m_limit_sss = 1e-22 # from 1610.08297
    ax1.loglog([m_limit_sss, m_limit_sss], [1e-14, 1e-9], '--', color='0.5')
    ax1.text(6e-23, 4e-11, s='small-scale structure', rotation=90)

    # 3G
    g_limit_yr = median_upper_lims_yr * (2*np.pi / 360) * 2 / 2.1e9 * \
                 (m_axion_yr/1e-21)# / np.sqrt(1.65)
    # sqrt(1.65) comes from ratio of uncertainties between 150s and 90s.
    # I almost definitely didn't calculate it right.
    ax1.loglog(m_axion_yr, g_limit_yr, '-C3')
    ax1.text(3.5e-21, 6e-12, s='SPT-3G 2019 forecast',
                 rotation=33, ha='center', color='C3')

    ax1.loglog(m_axion_yr, g_limit_yr / np.sqrt(5), '--C3')
    ax1.text(4e-21, 4.2e-12, s='SPT-3G full-survey forecast',
                 rotation=33, ha='center', color='C3')

    ax1.loglog(m_axion_yr, g_limit_yr / np.sqrt(5) / np.sqrt(30), '-.C3')
    ax1.text(8e-21, 2.5e-12, s='Rough CMB-S4 full-survey forecast',
                 rotation=33, ha='center', color='C3')

    ax1.axis([1e-23, 1e-18, 1e-14, 1e-9])
    ax1.set_xlabel('$m_\phi$ [eV]')
    ax1.set_ylabel('$g_{\phi\gamma}$ [GeV$^{-1}$]')

    def tick_fn(min_mass, max_mass):
        mass_per_freq = 2*np.pi * 6.528e-16 / (60*60*24)
        max_freq = max_mass / mass_per_freq
        min_freq = min_mass / mass_per_freq
        min_period = 1 / max_freq
        max_period = 1 / min_freq
        return min_period, max_period

    ax2 = ax1.twiny()
    ax2.set_xlim(ax1.get_xlim())
    min_p, max_p = tick_fn(1e-23, 1e-18)
    ax2.axis([max_p, min_p, 1e-14, 1e-9])
    ax2.set_xscale('log')
    ax2.set_xlabel('Oscillation period (days)')

    plt.show()

    '''
    #fig.tight_layout()
    if t_used:
        fig.savefig('/home/kferguson/axion_reordering_test/000_with_t_axion_forecast_%sGHz.png'%band, dpi=200)
    else:
        fig.savefig('/home/kferguson/axion_reordering_test/000_axion_forecast_%sGHz.png'%band, dpi=200)
    '''
