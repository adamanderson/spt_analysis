import numpy as np
from scipy.optimize import minimize


def neg2logL_time(A, phase, period, times, angle, angle_error):
    return np.sum((angle - A*np.sin(2*np.pi / period * times + phase))**2 / \
                      (angle_error**2))


def neg2logL_global_fit(period, times, angle, angle_error):
    def neg2logL_time_to_minimize(x, period):
        A = x[0]
        phase = x[1]
        return neg2logL_time(A, phase, period, times, angle, angle_error)
        
    out = minimize(neg2logL_time_to_minimize, (0.05, np.pi), args=(period), method='Powell')
    return out.fun, out.x


def neg2logL_profiled(A, period, times, angle, angle_error):
    # need to flip argument list
    def neg2logL_to_profile(phase, A, period):
        return neg2logL_time(A, phase, period, times, angle, angle_error)
    
    out = minimize(neg2logL_to_profile, np.pi, args=(A, period), method='Powell')
    return out.fun, out.x

    
def test_stat(A, period, times, angle, angle_error):
    neg2logL_global_fval, neg2logL_global_params = neg2logL_global_fit(period, times, angle, angle_error)
    neg2logL_profiled_fval, neg2logL_profiled_params = neg2logL_profiled(A, period, times, angle, angle_error)
    return neg2logL_profiled_fval - neg2logL_global_fval
