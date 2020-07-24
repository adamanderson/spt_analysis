# Simple script for computing the std's of scale pixel values, and performing a
# KS test to check the distribution for gaussianity.

import os.path
from spt3g import maps, core
from glob import glob
import numpy as np
import pickle
import re 
from scipy.stats import kstest

factor = {}
ks_test = {}

coadd_dir = '/sptlocal/user/kferguson/full_daniel_maps/'
coadd_fnames = glob(os.path.join(coadd_dir, '*.g3.gz'))

for fname in coadd_fnames:
    result = re.match('(.*?)_(.*?)GHz_(.*?)_map.g3.gz', os.path.basename(fname))
    field  = result.group(1)
    band   = int(result.group(2))
    obsid  = int(result.group(3))

    if field not in factor:
        factor[field] = {}
    if field not in ks_test:
        ks_test[field] = {}
    
    if obsid not in factor[field]:
        factor[field][obsid] = {}
    if obsid not in ks_test[field]:
        ks_test[field][obsid] = {}


    real_data = list(core.G3File(fname))[0]
    maps.RemoveWeights(real_data)

    q_arr = np.array(real_data['Q'])
    q_weight = np.array(real_data['Wpol'].QQ)

    q_arr_finite = q_arr[np.isfinite(q_arr)]
    q_weight_finite = q_weight[np.isfinite(q_arr)]

    factor[field][obsid][band] = np.std(q_arr_finite * np.sqrt(q_weight_finite))

    ks_result = kstest(q_arr_finite * np.sqrt(q_weight_finite) / factor[field][obsid][band],
                       cdf='norm')
    ks_test[field][obsid][band] = ks_result

    with open('weights_var_factors.pkl', 'wb') as f:
        save_dict = {'std_factor': factor,
                     'ks_result': ks_test}
        pickle.dump(save_dict, f)
