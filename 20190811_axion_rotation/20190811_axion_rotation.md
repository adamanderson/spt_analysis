---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3 (v2)
    language: python
    name: python3-v2
---

```python
import numpy as np
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
from scipy.optimize import minimize, fsolve, brentq
from scipy.stats import chi2, sigmaclip
```

```python
#Set up a new set of parameters for CAMB
pars = camb.CAMBparams()
#This function sets up CosmoMC-like settings, with one massive neutrino and helium set using BBN consistency
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
pars.set_for_lmax(2500, lens_potential_accuracy=0);

#calculate results for these parameters
results = camb.get_results(pars)

#get dictionary of CAMB power spectra
powers =results.get_cmb_power_spectra(pars, CMB_unit='muK')
for name in powers: print(name)
```

```python
#plot the total lensed CMB power spectra versus unlensed, and fractional difference
totCL=powers['total']
unlensedCL=powers['unlensed_scalar']
print(totCL.shape)
#Python CL arrays are all zero based (starting at L=0), Note L=0,1 entries will be zero by default.
#The different CL are always in the order TT, EE, BB, TE (with BB=0 for unlensed scalar results).
ls = np.arange(totCL.shape[0])
fig, ax = plt.subplots(3,2, figsize = (12,12))
ax[0,0].plot(ls,totCL[:,0], color='k')
ax[0,0].plot(ls,unlensedCL[:,0], color='r')
ax[0,0].set_title('TT')
ax[0,1].plot(ls[2:], 1-unlensedCL[2:,0]/totCL[2:,0]);
ax[0,1].set_title(r'$\Delta TT$')
ax[1,0].plot(ls,totCL[:,1], color='k')
ax[1,0].plot(ls,unlensedCL[:,1], color='r')
ax[1,0].set_title(r'$EE$')
ax[1,1].plot(ls,totCL[:,3], color='k')
ax[1,1].plot(ls,unlensedCL[:,3], color='r')
ax[1,1].set_title(r'$TE$');
ax[2,0].plot(ls,totCL[:,2], color='k')
ax[2,0].plot(ls,unlensedCL[:,2], color='r')
ax[2,0].set_title(r'$BB$')
ax[2,1].plot(ls,totCL[:,3], color='k')
ax[2,1].plot(ls,unlensedCL[:,3], color='r')
ax[2,1].set_title(r'$TE$');
for ax in ax.reshape(-1): ax.set_xlim([2,2500]);
```

```python
ells = ls[1:]
Cl_TT_full = unlensedCL[1:,0]/(ells*(ells+1)/(2*np.pi))
Cl_EE_full = unlensedCL[1:,1]/(ells*(ells+1)/(2*np.pi))
Cl_BB_full = unlensedCL[1:,2]/(ells*(ells+1)/(2*np.pi))
Cl_TE_full = unlensedCL[1:,3]/(ells*(ells+1)/(2*np.pi))

def generate_Cl(spectrum, f_sky, lmin, lmax, map_noise_T, beam_fwhm):
    l_range = ells[(ells>lmin) & (ells<lmax)]
    Cl_TT = Cl_TT_full[(ells>lmin) & (ells<lmax)]
    Cl_EE = Cl_EE_full[(ells>lmin) & (ells<lmax)]
    Cl_BB = Cl_BB_full[(ells>lmin) & (ells<lmax)]
    Cl_TE = Cl_TE_full[(ells>lmin) & (ells<lmax)]
    
    noise_weight_T = (map_noise_T * np.pi / 10800)**2
    noise_weight_P = (map_noise_T*np.sqrt(2.) * np.pi / 10800)**2
    sigma_beam = beam_fwhm / np.sqrt(8*np.log(2.)) * np.pi / 10800
    beam_weight = np.exp(-1.*l_range**2. * sigma_beam**2.)
    
    if spectrum == 'TT':
        var_factor = (Cl_TT + noise_weight_T * beam_weight)**2
        Cl_mean = Cl_TT
    elif spectrum == 'EE':
        var_factor = (Cl_EE + noise_weight_P * beam_weight)**2
        Cl_mean = Cl_EE
    elif spectrum == 'BB':
        var_factor = (Cl_BB + noise_weight_P * beam_weight)**2
        Cl_mean = Cl_BB
    elif spectrum == 'TE':
        var_factor = (Cl_TE**2 + \
                      (Cl_TT + noise_weight_T * beam_weight) * \
                      (Cl_EE + noise_weight_P * beam_weight))
        Cl_mean = Cl_TE
    elif spectrum == 'EB':
        var_factor = ((Cl_BB + noise_weight_P * beam_weight) * \
                      (Cl_EE + noise_weight_P * beam_weight))
        Cl_mean = np.zeros(Cl_TT.shape)
    elif spectrum == 'TB':
        var_factor = ((Cl_TT + noise_weight_T * beam_weight) * \
                      (Cl_BB + noise_weight_P * beam_weight))
        Cl_mean = np.zeros(Cl_TT.shape)
    
    Cl_sigma = np.sqrt(2 / ((2*l_range + 1)*f_sky) * var_factor)
    
    # generate a gaussian realization of all the Cl's, then
    # average them all into a single bin over the ell range
    # specified in the argument
    Cl_realization = np.random.normal(loc=Cl_mean, scale=Cl_sigma)
    
    return l_range, Cl_realization


def generate_Cl_averaged(spectrum, f_sky, lmin, lmax, map_noise_T, beam_fwhm):
    l_range, Cl_realization = generate_Cl(spectrum, f_sky, lmin, lmax,
                                          map_noise_T, beam_fwhm)
    Cl_averaged = np.mean(Cl_realization)
    return Cl_averaged
```

```python
Cl_avg_TB = generate_Cl_averaged('TB', f_sky=500/41000,
                                 lmin=100, lmax=3000,
                                 map_noise_T=6, beam_fwhm=1.2)
Cl_avg_EB = generate_Cl_averaged('EB', f_sky=500/41000,
                                 lmin=100, lmax=3000,
                                 map_noise_T=6, beam_fwhm=1.2)

print(Cl_avg_TB)
print(Cl_avg_EB)
```

```python
l_range, Cl_random = generate_Cl('TT', f_sky=500/41000,
                                 lmin=10, lmax=3000,
                                 map_noise_T=6, beam_fwhm=1.2)
plt.semilogy(l_range, Cl_random*(l_range*(l_range+1)/(2*np.pi)))
plt.semilogy(ells, Cl_TT_full*(ells*(ells+1)/(2*np.pi)))
```

```python
nsims = 1000
Cl_TB_random = np.zeros(nsims)
Cl_TE_random = np.zeros(nsims)
Cl_EB_random = np.zeros(nsims)
Cl_EE_random = np.zeros(nsims)
Cl_BB_random = np.zeros(nsims)
D_TB = np.zeros(nsims)
for jsim in range(nsims):
    Cl_TB_random[jsim] = generate_Cl_averaged('TB', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_TE_random[jsim] = generate_Cl_averaged('TE', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_EB_random[jsim] = generate_Cl_averaged('EB', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_EE_random[jsim] = generate_Cl_averaged('EE', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_BB_random[jsim] = generate_Cl_averaged('BB', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
_ = plt.hist(Cl_TB_random)
_ = plt.hist(Cl_EB_random)
print(np.std(Cl_TB_random))
print(np.std(Cl_EB_random))
```

Recall the $D$ quantities used in 0811.0618:
\begin{align}
D^{TB}_\ell &= C^{TB}_\ell \cos(2\alpha) - C^{TE}_\ell \sin(2\alpha) \\
D^{EB}_\ell &= C^{EB}_\ell \cos(4\alpha) - \frac{1}{2}\left( C^{EE}_\ell - C^{BB}_\ell \right) \sin(4\alpha).
\end{align}

Let's calculate the covariance under the null hypothesis that $\alpha = 0$. In this case, we have (dropping $\ell$ subscript because we have averaged over all $\ell$)
\begin{align}
D^{TB} &= C^{TB} \\
D^{EB} &= C^{EB},
\end{align}
so the variance of these two quantities form the diagonal elements of the covariance matrix.

```python
def pol_chi2(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB):
    if not isinstance(x, np.ndarray):
        alpha_array = np.array([x])
    else:
        alpha_array = x
    chi2_array = []
    for alpha in alpha_array:
        Dl_TB = Cl_TB * np.cos(2*alpha) - Cl_TE * np.sin(2*alpha)
        Dl_EB = Cl_EB * np.cos(4*alpha) - (1./2.) * (Cl_EE - Cl_BB) * np.sin(4*alpha)
        data = np.array([Dl_TB, Dl_EB]) 
        chi2_array.append(Dl_TB**2 / var_Dl_TB + Dl_EB**2 / var_Dl_EB)
    chi2_array = np.array(chi2_array)
    return chi2_array

def delta_chi2_plus1(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB, func_min):
    return pol_chi2(x, Cl_TB, Cl_TE, Cl_EB, Cl_EE, Cl_BB, var_Dl_TB, var_Dl_EB) - func_min - 1
```

```python
nsim_ul = 2000
delta_chi2_sims = np.zeros(nsim_ul)
angle_fit = np.zeros(nsim_ul)
angle_up1sigma = np.zeros(nsim_ul)
angle_down1sigma = np.zeros(nsim_ul)

Cl_TB_random = np.zeros(nsim_ul)
Cl_TE_random = np.zeros(nsim_ul)
Cl_EB_random = np.zeros(nsim_ul)
Cl_EE_random = np.zeros(nsim_ul)
Cl_BB_random = np.zeros(nsim_ul)
Dl_sims = np.zeros(nsim_ul)

for jsim in range(nsim_ul):
    Cl_TB_random[jsim] = generate_Cl_averaged('TB', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_TE_random[jsim] = generate_Cl_averaged('TE', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_EB_random[jsim] = generate_Cl_averaged('EB', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_EE_random[jsim] = generate_Cl_averaged('EE', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    Cl_BB_random[jsim] = generate_Cl_averaged('BB', f_sky=500/41000,
                                           lmin=100, lmax=3000,
                                           map_noise_T=6, beam_fwhm=1.2)
    
for jsim in range(nsim_ul):
    out = minimize(pol_chi2, 0, args=(Cl_TB_random[jsim], Cl_TE_random[jsim],
                                  Cl_EB_random[jsim], Cl_EE_random[jsim],
                                  Cl_BB_random[jsim],
                                  np.var(Cl_TB_random), np.var(Cl_EB_random)), method='Powell')
    angle_fit[jsim] = out.x
    Dl_sims[jsim] = out.fun
    
    delta_chi2_sims[jsim] = pol_chi2(0, Cl_TB_random[jsim], Cl_TE_random[jsim],
                                     Cl_EB_random[jsim], Cl_EE_random[jsim],
                                     Cl_BB_random[jsim],
                                     np.var(Cl_TB_random), np.var(Cl_EB_random)) - Dl_sims[jsim]
#     angle_interval[jsim] = fsolve(delta_chi2_plus1, x0=angle_fit[jsim],
#                                   args=(Dl_sims[jsim], Cl_TB_random[jsim], Cl_TE_random[jsim],
#                                   Cl_EB_random[jsim], Cl_EE_random[jsim], Cl_BB_random[jsim],
#                                   np.var(Cl_TB_random), np.var(Cl_EB_random)))
    angle_up1sigma[jsim] = newton(delta_chi2_plus1, x0=angle_fit[jsim] + 0.01,
                                  args=(Cl_TB_random[jsim], Cl_TE_random[jsim],
                                  Cl_EB_random[jsim], Cl_EE_random[jsim], Cl_BB_random[jsim],
                                  np.var(Cl_TB_random), np.var(Cl_EB_random), Dl_sims[jsim]))
    angle_down1sigma[jsim] = newton(delta_chi2_plus1, x0=angle_fit[jsim] - 0.01,
                                  args=(Cl_TB_random[jsim], Cl_TE_random[jsim],
                                  Cl_EB_random[jsim], Cl_EE_random[jsim], Cl_BB_random[jsim],
                                  np.var(Cl_TB_random), np.var(Cl_EB_random), Dl_sims[jsim]))
```

```python
_ = plt.hist(angle_up1sigma, histtype='step')
_ = plt.hist(angle_down1sigma, histtype='step')
_ = plt.hist(angle_up1sigma - angle_down1sigma, histtype='step')
```

```python
_ = plt.hist(delta_chi2_sims, bins=np.linspace(0,20,101), normed=True)
plt.plot(np.linspace(0, 20, 1000), chi2.pdf(np.linspace(0, 20, 1000), df=1))
plt.gca().set_yscale('log')
# plt.ylim([1e-3, 1])
```

## Full time-domain simulation
Let's consider the case of an idealized SPTpol dataset. The white noise level of the Henning E-mode analysis is quoted as 9.4 $\mu$K-arcmin in polarization, after cuts. Let's assume that this weight is distributed across all observations, and that all observations have uniform map coverage. The exact number of observations is a little confusing, but the number 3491 is used. The per-observation noise level is therefore:

```python
total_depth = 9.4 # [uK arcmin] at 150 GHz
n_observations = 3491
observations_per_bundle = 40
n_bundles = int(np.floor(n_observations / observations_per_bundle))
noise_per_observation = total_depth * np.sqrt(n_observations)
noise_per_bundle = total_depth * np.sqrt(n_bundles)
```

```python
noise_per_bundle
```

Next, let's generate fake $C_\ell$ for each observation including instrumental noise.

```python
Cl_TB_per_obs = np.zeros(n_bundles)
Cl_TE_per_obs = np.zeros(n_bundles)
Cl_EB_per_obs = np.zeros(n_bundles)
Cl_EE_per_obs = np.zeros(n_bundles)
Cl_BB_per_obs = np.zeros(n_bundles)


for jsim in np.arange(n_bundles):
    Cl_TB_per_obs[jsim] = generate_Cl_averaged('TB', f_sky=500/41000,
                                               lmin=100, lmax=3000,
                                               map_noise_T=noise_per_bundle,
                                               beam_fwhm=1.2)
    Cl_TE_per_obs[jsim] = generate_Cl_averaged('TE', f_sky=500/41000,
                                               lmin=100, lmax=3000,
                                               map_noise_T=noise_per_bundle,
                                               beam_fwhm=1.2)
    Cl_EB_per_obs[jsim] = generate_Cl_averaged('EB', f_sky=500/41000,
                                               lmin=100, lmax=3000,
                                               map_noise_T=noise_per_bundle,
                                               beam_fwhm=1.2)
    Cl_EE_per_obs[jsim] = generate_Cl_averaged('EE', f_sky=500/41000,
                                               lmin=100, lmax=3000,
                                               map_noise_T=noise_per_bundle,
                                               beam_fwhm=1.2)
    Cl_BB_per_obs[jsim] = generate_Cl_averaged('BB', f_sky=500/41000,
                                               lmin=100, lmax=3000,
                                               map_noise_T=noise_per_bundle,
                                               beam_fwhm=1.2)
```

Next, we use our fake $C_\ell$ to estimate the polarization angle from each observation using the $D$ estimator.

```python
alpha_per_obs = np.zeros(n_bundles)
alpha_up1sigma = np.zeros(n_bundles)
alpha_down1sigma = np.zeros(n_bundles)
Dl_per_obs = np.zeros(n_bundles)

for jsim in np.arange(n_bundles):
    out = minimize(pol_chi2, 0, args=(Cl_TB_per_obs[jsim], Cl_TE_per_obs[jsim],
                                      Cl_EB_per_obs[jsim], Cl_EE_per_obs[jsim],
                                      Cl_BB_per_obs[jsim],
                                      np.var(Cl_TB_per_obs), np.var(Cl_EB_per_obs)), method='Powell')
    alpha_per_obs[jsim] = out.x
    Dl_per_obs[jsim] = out.fun
    
    alpha_up1sigma[jsim] = newton(delta_chi2_plus1, x0=alpha_per_obs[jsim] + 0.01,
                                  args=(Cl_TB_per_obs[jsim], Cl_TE_per_obs[jsim],
                                  Cl_EB_per_obs[jsim], Cl_EE_per_obs[jsim], Cl_BB_per_obs[jsim],
                                  np.var(Cl_TB_per_obs), np.var(Cl_EB_per_obs), Dl_per_obs[jsim]))
    alpha_down1sigma[jsim] = newton(delta_chi2_plus1, x0=alpha_per_obs[jsim] - 0.01,
                                  args=(Cl_TB_per_obs[jsim], Cl_TE_per_obs[jsim],
                                  Cl_EB_per_obs[jsim], Cl_EE_per_obs[jsim], Cl_BB_per_obs[jsim],
                                  np.var(Cl_TB_per_obs), np.var(Cl_EB_per_obs), Dl_per_obs[jsim]))
    
alpha_error_per_obs = (alpha_up1sigma - alpha_down1sigma) / 2

# The minimizer sometimes fails and best fit ends up at N*pi/4 away from zero.
# For now, just throw these values away.
# TODO: Do something sensible with these values.
alpha_per_obs_clipped, clip_lower, clip_upper = sigmaclip(alpha_per_obs, low=3, high=3)
alpha_error_per_obs_clipped = alpha_error_per_obs[(alpha_per_obs>clip_lower) & \
                                                  (alpha_per_obs<clip_upper)]
alpha_per_obs = alpha_per_obs_clipped
alpha_error_per_obs = alpha_error_per_obs_clipped
```

```python
_ = plt.hist(alpha_per_obs, bins=np.linspace(-0.2, 0.2))
```

```python
_ = plt.hist(alpha_error_per_obs, bins=np.linspace(-0.05,0.05))
```

```python
_ = plt.errorbar(np.arange(len(alpha_per_obs)), alpha_per_obs * 180/np.pi,
                 yerr=alpha_error_per_obs * 180/np.pi,
                 linestyle='None', marker='o', markersize=3)
plt.xlabel('bundle index')
plt.ylabel('rotation angle [deg]')
plt.title('simulated global rotation angle\n'
          '{} observations / bundle, {:.1f} uK-arcmin per bundle\n'
          '(Henning, et al. (2017) sensitivity)'.format(observations_per_bundle, noise_per_bundle))
plt.tight_layout()
plt.savefig('sim_1x_rotation_angle_vs_time.png', dpi=100)
```

```python
time_per_obs = np.arange(len(alpha_per_obs)) # time is currently bundle index

def neg2logL_time(A, phase, period):
    return np.sum((alpha_per_obs - A*np.sin(2*np.pi / period * time_per_obs + phase))**2 / \
                      (alpha_error_per_obs**2))

def neg2logL_global_fit(period):
    def neg2logL_time_to_minimize(x, period):
        A = x[0]
        phase = x[1]
        return neg2logL_time(A, phase, period)
        
    out = minimize(neg2logL_time_to_minimize, (0.05, np.pi), args=(period), method='Powell')
    return out.fun, out.x

def neg2logL_profiled(A, period):
    # need to flip argument list
    def neg2logL_to_profile(phase, A, period):
        return neg2logL_time(A, phase, period)
    
    out = minimize(neg2logL_to_profile, np.pi, args=(A, period), method='Powell')
    return out.fun, out.x
    
def test_stat(A, period):
    neg2logL_global_fval, neg2logL_global_params = neg2logL_global_fit(period)
    neg2logL_profiled_fval, neg2logL_profiled_params = neg2logL_profiled(A, period)
    return neg2logL_profiled_fval - neg2logL_global_fval
```

```python
periods_to_test = np.arange(4, 80)
amplitude_best_fit = np.zeros(len(periods_to_test))
for jperiod, period in enumerate(periods_to_test):
    amplitudes = np.linspace(0, 2e-2)
    test_stats = np.array([test_stat(A, period) for A in amplitudes])
    neg2logL_global_fval, neg2logL_global_params = neg2logL_global_fit(period)
    amplitude_best_fit[jperiod] = neg2logL_global_params[0]
```

```python
plt.plot(periods_to_test, amplitude_best_fit * 180 / np.pi)
plt.xlabel('oscillation period [# of bundles]')
plt.ylabel('oscillation amplitude [deg]')
plt.title('best-fit oscillation amplitude (single realization)')
plt.tight_layout()
plt.savefig('sim_1x_bestfit_A_vs_period.png', dpi=100)
```

```python
neg2logL_global_fval, neg2logL_global_params = neg2logL_global_fit(period)

_ = plt.errorbar(np.arange(len(alpha_per_obs)), alpha_per_obs, yerr=alpha_error_per_obs,
                 linestyle='None', marker='o', markersize=3)

time_per_obs = np.arange(len(alpha_per_obs))
plt.plot(time_per_obs, neg2logL_global_params[0]*\
         np.sin(2*np.pi / period * time_per_obs + neg2logL_global_params[1]))
plt.ylim([-0.1, 0.1])
plt.xlabel('oscillation period [# of bundles]')
plt.ylabel('oscillation amplitude [deg]')
plt.title('best-fit oscillation amplitude (period = {})'.format(period))
plt.tight_layout()
plt.savefig('sim_1x_bestfit_example.png', dpi=100)
```

```python
neg2logL_global_params[0]
```

```python
neg2logL_global_fval
```

```python
0.01 * 180 / np.pi
```

```python

```
