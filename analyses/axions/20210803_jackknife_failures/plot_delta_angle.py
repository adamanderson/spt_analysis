import matplotlib.pyplot as plt
import numpy as np
from spt3g import core
from glob import glob

fname_pattern = '/sptlocal/user/kferguson/updated_noise_jackknife_amp_dists_v2/collated_split_best_fit_amps_150GHz_ra0hdec*'
fnames = glob(fname_pattern)

for fname in fnames:
    data = list(core.G3File(fname))[0]
    plt.hist()