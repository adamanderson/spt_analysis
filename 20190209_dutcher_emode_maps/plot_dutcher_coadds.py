# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, dfmux, calibration, mapmaker, mapspectra
from spt3g.mapmaker.mapmakerutils import remove_weight

bands = [90, 150, 220]
directions = ['Left', 'Right']
stokes_params = ['Q', 'U', 'T']
plt.bone()

d = [fr for fr in core.G3File('/spt/user/ddutcher/coadds/coadd2018_noW201wafCMpoly19mhpf300.g3')]

maps = {}
for fr in d:
    for direc in directions:
        if direc in fr['Id']:
            direction = direc
    for b in bands:
        if str(b) in fr['Id']:
            band = b

    if band not in maps.keys():
        maps[band] = {}
    
    #T, Q, U = remove_weight(fr['T'], fr['Q'], fr['U'], fr['Wpol'])
    maps[band][direction] = {'T':fr['T'], 'Q':fr['Q'], 'U':fr['U'], 'Wpol':fr['Wpol']}


for band in [90, 150, 220]:

    # L+R
    for combo in ['sum', 'diff']:
        if combo == 'sum':
            T, Q, U = remove_weight(maps[band]['Left']['T'] + maps[band]['Right']['T'],
                                    maps[band]['Left']['Q'] + maps[band]['Right']['Q'],
                                    maps[band]['Left']['U'] + maps[band]['Right']['U'],
                                    maps[band]['Left']['Wpol'] + maps[band]['Right']['Wpol'])
        elif combo == 'diff':
            T, Q, U = remove_weight(maps[band]['Left']['T'] - maps[band]['Right']['T'],
                                    maps[band]['Left']['Q'] - maps[band]['Right']['Q'],
                                    maps[band]['Left']['U'] - maps[band]['Right']['U'],
                                    maps[band]['Left']['Wpol'] + maps[band]['Right']['Wpol'])
        apod = mapspectra.apodmask.makeBorderApodization(
            maps[band]['Left']['Wpol'] + maps[band]['Right']['Wpol'],
            apod_type='cos', radius_arcmin=50.,
            zero_border_arcmin=5, smooth_weights_arcmin=5,
            use_square_smoothing=True)


        Qmap = Q * apod / core.G3Units.microkelvin
        Q_masked = np.ma.array(Qmap, mask=(Qmap==0))
        plt.figure(figsize=(8,5))
        plt.imshow(Q_masked, vmin=-25, vmax=25, interpolation='gaussian')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('Q_{}_{}.png'.format(band, combo), dpi=400)
        plt.close()


        Umap = U * apod / core.G3Units.microkelvin
        U_masked = np.ma.array(Umap, mask=(Umap==0))
        plt.figure(figsize=(8,5))
        plt.imshow(U_masked, vmin=-25, vmax=25, interpolation='gaussian')
        plt.colorbar()
        plt.axis('off')
        plt.tight_layout()
        plt.savefig('U_{}_{}.png'.format(band, combo), dpi=400)
        plt.close()


