import numpy as np 
import pickle
from glob import glob
import os.path

datadir = '/sptgrid/user/adama/20210407_multiband_limits/test_limits_bkg_only_with_bayesianfit_with_chi2'
fnames = np.sort(glob(os.path.join(datadir, 'sim_bkg_only_fitfreq=0.01000-2.00000_amp=0.00000_freq=0.01000_*.pkl')))

summary_dict = {'A_upperlimit':{},
                'A_fit':{},
                'chi2(A=A_fit)':{},
                'chi2(A=0)':[]}
for fn in fnames[:100]:
    print(fn)

    with open(fn, 'rb') as f:
        d = pickle.load(f)

    for j in d:
        for k in summary_dict:
            if type(summary_dict[k]) is dict:
                for freq in d[j][k].keys():
                    if freq not in summary_dict[k]:
                        summary_dict[k][freq] = []
                    summary_dict[k][freq].append(d[j][k][freq])
            elif type(summary_dict[k]) is list:
                summary_dict[k].append(d[j][k])

for k in summary_dict:
    if type(summary_dict[k]) is dict:
        for freq in d[j][k].keys():
            summary_dict[k][freq] = np.array(summary_dict[k][freq])
    elif type(summary_dict[k]) is list:
        summary_dict[k] = np.array(summary_dict[k])

with open('collated_bgonly_sims.pkl', 'wb') as f:
    pickle.dump(summary_dict, f)