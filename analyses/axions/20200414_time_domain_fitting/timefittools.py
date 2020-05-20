import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize

def time_domain_model(amp, freq, phase, times):
    return amp*np.sin(2*np.pi*freq*times + phase)

def minimize_chi2(data):
    func = lambda x: chi2(x[0], x[1], x[2], data)
    result = minimize(func, x0=(0.1, 0.1, np.pi), method='Powell')
    return result['fun']

def chi2(amp, freq, phase, data): 
    times = data['times']
    pol_angles = data['angles']
    pol_angle_errs = data['errs']
    return np.sum((pol_angles - time_domain_model(amp, freq, phase, times))**2 / pol_angle_errs**2)

def posterior(amp, freq, phase, data, chi2_min):
    return np.exp(-1*(chi2(amp, freq, phase, data) - chi2_min) / 2)

def posterior_marginal(amp, freq, data, chi2_min):
    P = lambda phase: posterior(amp, freq, phase, data, chi2_min)
    result, abserr = quad(P, 0, 2*np.pi)
    return result

def posterior_marginal_cdf(amp, freq, data, chi2_min):
    amp_max = 10 # hard-coded maximum amplitude to use for normalization

    P = lambda A: posterior_marginal(A, freq, data, chi2_min) 
    cdf_max, _ = quad(P, 0, amp_max)
    cdf, _ = quad(P, 0, amp)
    cdf = cdf / cdf_max
    return cdf

def upper_limit_bayesian(freq, data, cl, amp_cl_range, chi2_min):
    cdf_minus_cl = lambda A: posterior_marginal_cdf(A, freq, data, chi2_min) - cl
    amp_cl = brentq(cdf_minus_cl, amp_cl_range[0], amp_cl_range[1])
    return amp_cl
