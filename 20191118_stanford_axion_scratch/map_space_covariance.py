# quick test to calculate the pixel-pixel covariance over all of Daniel's 2018 EE maps
import numpy as np
from spt3g import core, mapmaker
from spt3g.mapmaker import remove_weight
from glob import glob
import pickle

# coadds data
fname_coadd = '/spt/user/ddutcher/coadds/20190917_full_150GHz.g3.gz'
d_coadd = list(core.G3File(fname_coadd))[0]
_, _, u_coadd_noweight = remove_weight(d_coadd['T'], d_coadd['Q'], d_coadd['U'], d_coadd['Wpol'])

# individual maps data
fname_maps = glob('/spt/user/ddutcher/ra0hdec-52.25/y1_ee_20190811/' + \
                  'high_150GHz_left_maps/*/high_150GHz_left_maps_*.g3.gz')

# pixel for which to calculate covariance
pixel = (850,1100)

running_correlation = np.zeros(u_coadd_noweight.shape)

n_maps = 0
for fname in fname_maps:
    print(fname)

    d_single = list(core.G3File(fname))[5]
    _, _, u_single_noweight = remove_weight(d_single['T'], d_single['Q'], d_single['U'], d_single['Wpol'])
    # u_residual_i0 = u_single_noweight[pixel[0], pixel[1]] - u_coadd_noweight[pixel[0], pixel[1]]
    # u_residual_j = u_single_noweight - u_coadd_noweight
    # u_residual_i0 = u_single_noweight[pixel[0], pixel[1]]
    # u_residual_j = u_single_noweight
    u_residual_i0 = d_single['U'][pixel[0], pixel[1]]
    u_residual_j = d_single['U']

    if np.isfinite(u_residual_i0):
        running_correlation += np.array(u_residual_i0 * u_residual_j)
        n_maps += 1

running_correlation /= n_maps

with open('correlation.pkl', 'wb') as f:
    pickle.dump(running_correlation, f)
