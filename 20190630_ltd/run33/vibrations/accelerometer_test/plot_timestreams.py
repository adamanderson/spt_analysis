import netCDF4
import matplotlib.pyplot as plt
from scipy.signal import welch
import numpy as np

d = netCDF4.Dataset('vibration_5_2V.nc')

for bolos in d.variables:
    if '0136.1.1' in bolos:
        plt.figure()
        ff, psd = welch(d[bolos][5000:], fs=152.5, nperseg=256)
        asd = np.sqrt(psd)
        plt.semilogy(ff, asd)
        plt.savefig('bolo_figures/{}_psd.png'.format(bolos))
        plt.close()

        plt.figure()
        plt.plot(d[bolos][5000:])
        plt.savefig('bolo_figures/{}_tod.png'.format(bolos))
        plt.close()
