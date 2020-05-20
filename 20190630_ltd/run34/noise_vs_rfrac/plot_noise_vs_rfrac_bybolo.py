import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tes_noise import *
from scipy.constants import Boltzmann as kB

Tbase = 350
fname_dark = '20181014_141329_noise_v_rfrac_hkdata.pkl'
fname_optical = '20181012_225752_noise_v_rfrac_hkdata.pkl'
fnames = [fname_dark, fname_optical]
noise_vs_rfrac_dark = dict()
noise_vs_rfrac_optical = dict()
noise_vs_rfrac = [noise_vs_rfrac_dark, noise_vs_rfrac_optical]
vbias_vs_rfrac_dark = dict()
vbias_vs_rfrac_optical = dict()
vbias_vs_rfrac = [vbias_vs_rfrac_dark, vbias_vs_rfrac_optical]
stub_name = 'Fnal_run34_w206_noiseVrfrac'
band_width = {90:25e9, 150:35e9, 220:45e9}
band_center = {90:90e9, 150:150e9, 220:220e9}
efficiency = 0.85
temp_coldload = 21.5
temp_coldload_cold = 3.5
# band_power = {90:kB*temp_coldload*band_width[90]*efficiency,
#               150:kB*temp_coldload*band_width[150]*efficiency,
#               220:kB*temp_coldload*band_width[220]*efficiency}
delta_Psat = {90: 4.8e-12, 150: 6.0e-12, 220: 5.4e-12}
band_power = {band: delta_Psat[band] * temp_coldload / (temp_coldload - temp_coldload_cold) for band in delta_Psat}
photon_noise = {90:np.sqrt(shot_noise(band_center[90], band_power[90])**2.0 + \
                           correlation_noise(band_center[90], band_power[90], band_width[90], 1.0)**2.0),
                150:np.sqrt(shot_noise(band_center[150], band_power[150])**2.0 + \
                           correlation_noise(band_center[150], band_power[150], band_width[150], 1.0)**2.0),
                220:np.sqrt(shot_noise(band_center[220], band_power[220])**2.0 + \
                           correlation_noise(band_center[220], band_power[220], band_width[220], 1.0)**2.0)}
band_dict = dict()

for fname, noise_dict, vbias_dict in zip(fnames, noise_vs_rfrac, vbias_vs_rfrac):
    with open(fname, 'rb') as f:
        d = pickle.load(f, encoding='latin1')

    for amp in np.sort(list(d.keys())):
        noise_fname = glob('{}/data/*pkl'.format(d[amp]))[0]

        with open(noise_fname, 'rb') as f:
            dnoise = pickle.load(f, encoding='latin1')

        pstrings = np.unique(['/'.join(bolo.split('/')[:3]) for bolo in dnoise])

        for chan in dnoise:
            boloname = dnoise[chan]['bolo_name']
            if boloname not in noise_dict:
                noise_dict[boloname] = dict()
            if boloname not in vbias_dict:
                vbias_dict[boloname] = dict()
            noise_dict[boloname][amp] = dnoise[chan]['noise']['i_phase']['median_noise']
            vbias_dict[boloname][amp] = dnoise[chan]['calc_R']['V']
            
            if boloname not in band_dict:
                if dnoise[chan]['physical_name'] != 'lc_resistor':
                    band_dict[boloname] = int(dnoise[chan]['physical_name'].split('.')[1])

# per bolo noise plots
if False:
    bolos_plot = np.intersect1d(list(noise_vs_rfrac_dark.keys()),
                                list(noise_vs_rfrac_optical.keys()))
    for bolo in bolos_plot:
        plt.figure()
        rfracs_dark = np.array([rfrac for rfrac in noise_vs_rfrac_dark[bolo]])
        noise_dark = np.array([noise_vs_rfrac_dark[bolo][rfrac] for rfrac in noise_vs_rfrac_dark[bolo]])
        sortind = np.argsort(rfracs_dark)
        rfracs_dark = rfracs_dark[sortind]
        noise_dark = noise_dark[sortind]

        rfracs_optical = np.array([rfrac for rfrac in noise_vs_rfrac_optical[bolo]])
        noise_optical = np.array([noise_vs_rfrac_optical[bolo][rfrac] for rfrac in noise_vs_rfrac_optical[bolo]])
        sortind = np.argsort(rfracs_optical)
        rfracs_optical = rfracs_optical[sortind]
        noise_optical = noise_optical[sortind]

        plt.plot(rfracs_dark, noise_dark, 'o-', label='dark')
        plt.plot(rfracs_optical, noise_optical, 'o-', label='21.5K load')
        plt.xlabel('rfrac')
        plt.ylabel('NEI [pA/rtHz]')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures_darkVoptical/{}_noiseVrfrac.png'.format(bolo), dpi=200)
        plt.close()

# noise distributions across bands
rfrac_list = [0.75, 0.80, 0.85, 0.90, 0.95]
noise_medians_dark = dict()
noise_up1sigma_dark = dict()
noise_down1sigma_dark = dict()
noise_medians_optical = dict()
noise_up1sigma_optical = dict()
noise_down1sigma_optical = dict()
nep_medians_dark = dict()
nep_up1sigma_dark = dict()
nep_down1sigma_dark = dict()
nep_medians_optical = dict()
nep_up1sigma_optical = dict()
nep_down1sigma_optical = dict()

for band in [90, 150, 220]:
    noise = dict()
    nep = dict()
    noise_medians_optical[band] = []
    noise_up1sigma_optical[band] = []
    noise_down1sigma_optical[band] = []
    nep_medians_optical[band] = []
    nep_up1sigma_optical[band] = []
    nep_down1sigma_optical[band] = []
    for rfrac in rfrac_list:
        noise[rfrac] = np.array([noise_vs_rfrac_optical[bolo][rfrac]
                                 for bolo in noise_vs_rfrac_optical
                                 if rfrac in noise_vs_rfrac_optical[bolo] and \
                                 bolo in band_dict and \
                                 band_dict[bolo] == band])
        nep[rfrac] = np.array([noise_vs_rfrac_optical[bolo][rfrac] * vbias_vs_rfrac_optical[bolo][rfrac] / 2.0 * 1e6
                                 for bolo in noise_vs_rfrac_optical
                                 if rfrac in noise_vs_rfrac_optical[bolo] and \
                                 bolo in band_dict and \
                                 band_dict[bolo] == band])
        noise_medians_optical[band].append(np.median(noise[rfrac]))
        noise_up1sigma_optical[band].append(np.percentile(noise[rfrac], 86))
        noise_down1sigma_optical[band].append(np.percentile(noise[rfrac], 14))
        nep_medians_optical[band].append(np.median(nep[rfrac]))
        nep_up1sigma_optical[band].append(np.percentile(nep[rfrac], 86))
        nep_down1sigma_optical[band].append(np.percentile(nep[rfrac], 14))

    noise = dict()
    nep = dict()
    noise_medians_dark[band] = []
    noise_up1sigma_dark[band] = []
    noise_down1sigma_dark[band] = []
    nep_medians_dark[band] = []
    nep_up1sigma_dark[band] = []
    nep_down1sigma_dark[band] = []
    for rfrac in rfrac_list:
        noise[rfrac] = np.array([noise_vs_rfrac_dark[bolo][rfrac]
                                 for bolo in noise_vs_rfrac_dark
                                 if rfrac in noise_vs_rfrac_dark[bolo] and \
                                 bolo in vbias_vs_rfrac_optical and \
                               rfrac in vbias_vs_rfrac_optical[bolo] and \
                                 band_dict[bolo] == band])
        nep[rfrac] = np.array([noise_vs_rfrac_dark[bolo][rfrac] * vbias_vs_rfrac_optical[bolo][rfrac] / 2.0 * 1e6
                                 for bolo in noise_vs_rfrac_dark
                                 if rfrac in noise_vs_rfrac_dark[bolo] and \
                               bolo in vbias_vs_rfrac_optical and \
                               rfrac in vbias_vs_rfrac_optical[bolo] and \
                                 band_dict[bolo] == band])
        noise_medians_dark[band].append(np.median(noise[rfrac]))
        noise_up1sigma_dark[band].append(np.percentile(noise[rfrac], 86))
        noise_down1sigma_dark[band].append(np.percentile(noise[rfrac], 14))
        nep_medians_dark[band].append(np.median(nep[rfrac]))
        nep_up1sigma_dark[band].append(np.percentile(nep[rfrac], 86))
        nep_down1sigma_dark[band].append(np.percentile(nep[rfrac], 14))


plt.figure()
for band in [90, 150, 220]:
    plt.errorbar(rfrac_list, noise_medians_dark[band],
                 yerr=[np.array(noise_medians_dark[band]) - np.array(noise_down1sigma_dark[band]),
                       np.array(noise_up1sigma_dark[band]) - np.array(noise_medians_dark[band])],
                 label='{} GHz'.format(band), marker='o')
plt.legend()
plt.title('dark')
plt.savefig('noiseVrfrac_median_dark.png', dpi=200)
plt.close()

plt.figure()
for band in [90, 150, 220]: 
    plt.errorbar(rfrac_list, noise_medians_optical[band],
                 yerr=[np.array(noise_medians_optical[band]) - np.array(noise_down1sigma_optical[band]),
                       np.array(noise_up1sigma_optical[band]) - np.array(noise_medians_optical[band])],
                 label='{} GHz'.format(band), marker='o')
plt.legend()
plt.title('optical / 21.5K load')
plt.savefig('noiseVrfrac_median_optical.png', dpi=200)

for band in [90, 150, 220]:
    plt.figure()
    plt.errorbar(rfrac_list, noise_medians_dark[band],
                 yerr=[np.array(noise_medians_dark[band]) - np.array(noise_down1sigma_dark[band]),
                       np.array(noise_up1sigma_dark[band]) - np.array(noise_medians_dark[band])],
                 label='dark', marker='o')
    plt.errorbar(rfrac_list, noise_medians_optical[band],
                 yerr=[np.array(noise_medians_optical[band]) - np.array(noise_down1sigma_optical[band]),
                       np.array(noise_up1sigma_optical[band]) - np.array(noise_medians_optical[band])],
                 label='21.5K load', marker='o')
    plt.legend()
    plt.title('{} GHz'.format(band))
    plt.xlabel('rfrac')
    plt.ylabel('NEI [pA/rtHz]')
    plt.tight_layout()
    plt.savefig('noiseVrfrac_median_darkVoptical_{}_Tbase={}.png'.format(band, Tbase), dpi=200)

for band in [90, 150, 220]:
    plt.figure()
    plt.errorbar(rfrac_list, nep_medians_dark[band],
                 yerr=[np.array(nep_medians_dark[band]) - np.array(nep_down1sigma_dark[band]),
                       np.array(nep_up1sigma_dark[band]) - np.array(nep_medians_dark[band])],
                 label='dark', marker='o')
    plt.errorbar(rfrac_list, nep_medians_optical[band],
                 yerr=[np.array(nep_medians_optical[band]) - np.array(nep_down1sigma_optical[band]),
                       np.array(nep_up1sigma_optical[band]) - np.array(nep_medians_optical[band])],
                 label='21.5K load', marker='o')
    plt.plot(rfrac_list, photon_noise[band]*1e18*np.ones(len(rfrac_list)), label='photon noise ({:.1f} pW)'.format(band_power[band]*1e12))
    photon = photon_noise[band]*1e18*np.ones(len(rfrac_list))
    dark = np.array(nep_medians_dark[band])
    plt.legend()
    plt.title('{} GHz'.format(band))
    plt.xlabel('rfrac')
    plt.ylabel('NEP [aW/rtHz]')
    plt.tight_layout()
    plt.savefig('nepVrfrac_median_darkVoptical_{}_Tbase={}.png'.format(band, Tbase), dpi=200)

