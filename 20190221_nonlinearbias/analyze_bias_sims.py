from spt3g import core, mapmaker
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle
import os.path
from glob import glob


fname_nobias = 'sim_cl_bias_none.pkl'
fname_bias = 'sim_cl_bias_linear_2.67.pkl'

with open(fname_nobias, 'rb') as f:
    d_nobias = pickle.load(f)

with open(fname_bias, 'rb') as f:
    d_bias = pickle.load(f)

# plot maps
for fname in glob('*g3'):
    dmap = [fr for fr in core.G3File(fname)]

    plt.figure(figsize=(12,6))
    plt.imshow(dmap[0]['T'], vmin=-700, vmax=700)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('figures/{}.png'.format(os.path.splitext(fname)[0]), dpi=200)
    plt.close()


# plot power spectra
for spectrum in ['TT', 'TE', 'EE', 'BB']:
    for sim_fname in d_nobias.keys():
        avg_cls_nobias = np.zeros(d_nobias[sim_fname]['ell'].shape)
        for sim_fname in d_nobias:
            avg_cls_nobias = avg_cls_nobias + d_nobias[sim_fname][spectrum]

        sim_fnames = list(d_bias.keys())
        avg_cls_bias = np.zeros(d_bias[sim_fname]['ell'].shape)
        for sim_fname in d_bias:
            avg_cls_bias = avg_cls_bias + d_bias[sim_fname][spectrum]

    ell_nobias = d_nobias[sim_fname]['ell']
    ell_bias = d_bias[sim_fname]['ell']

    # renormalize the power spectrum in 500 < ell < 1500
    ell_range = (ell_nobias > 500) & (ell_nobias < 1500)
    bias_normalization = np.mean(avg_cls_bias[ell_range] / avg_cls_nobias[ell_range])

    # plotting
    plt.figure()
    gs1 = GridSpec(5,1)
    gs1.update(wspace=0.0, hspace=0.0)

    ax1 = plt.subplot(gs1[:4, :])
    plt.plot(ell_nobias, avg_cls_nobias * ((ell_nobias * (ell_nobias+1)) / (2.*np.pi)),
             'o-', color='C0', label='no bias', markersize=3, linewidth=1)
    plt.plot(ell_bias, avg_cls_bias * ((ell_nobias * (ell_nobias+1)) / (2.*np.pi)) / bias_normalization,
             'o-', fillstyle='none', color='C1', label='linear bias', markersize=3, linewidth=1)
    plt.grid()
    plt.legend()
    plt.ylabel('D_\ell^{} [\mu K^2]'.format(spectrum))

    ax2 = plt.subplot(gs1[4, :])
    plt.plot(ell_nobias, avg_cls_bias/avg_cls_nobias / bias_normalization)
    plt.grid()
    plt.xlabel('\mathcal{\ell}')
    plt.ylabel('[linear bias] / [no bias]')
    plt.savefig('figures/{}_linear_bias.png'.format(spectrum), dpi=200)
    plt.close()
