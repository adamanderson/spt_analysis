import pickle
import numpy as np
import matplotlib.pyplot as plt
import pydfmux.analysis.utils.pixel_tools as pxtools
import pydfmux

dark_pixels = [271, 250, 225, 196, 163]

with open('/home/adama/hardware_maps/fnal/run33/hwm.yaml', 'rb') as f:
    hwm = pydfmux.load_session(f)['hardware_map']

with open('fnal_run33_slots12_0135_0136_optical_eff_fit_results.pkl', 'rb') as f:
    d = pickle.load(f)

slopes_optical = dict()
slopes_optical_corrected = dict()
efficiency_optical = dict()
efficiency_optical_corrected = dict()
freqs_optical = dict()
slopes_blanked = dict()
slopes_blanked_outer = dict()
slopes_dark = dict()
median_slopes_blanked_outer = dict()
for band in [90, 150, 220]:
    slopes_optical[band] = np.array([d[bolo]['slope_CL'] for bolo in d
                                     if str(d[bolo]['pstring']).split('/')[0] == '0135' and
                                        str(d[bolo]['pstring']).split('/')[1] == '2' and
                                        bolo.split('.')[1] == str(band)])
    efficiency_optical[band] = np.array([d[bolo]['opteff_CL'] for bolo in d
                                     if str(d[bolo]['pstring']).split('/')[0] == '0135' and
                                        str(d[bolo]['pstring']).split('/')[1] == '2' and
                                        bolo.split('.')[1] == str(band)])
    freqs_optical[band] = np.array([hwm.channel_maps_from_pstring(bolo)[0].lc_channel.frequency for bolo in d
                                     if str(d[bolo]['pstring']).split('/')[0] == '0135' and
                                        str(d[bolo]['pstring']).split('/')[1] == '2' and
                                        bolo.split('.')[1] == str(band)])
    slopes_blanked[band] = np.array([d[bolo]['slope_CL'] for bolo in d
                                     if str(d[bolo]['pstring']).split('/')[0] == '0136' and
                                        bolo.split('.')[1] == str(band)])
    slopes_dark[band] = np.array([d[bolo]['slope_CL'] for bolo in d
                                     if bolo.split('.')[1] == str(band) and
                                     int(bolo.split('.')[0].split('/')[1]) in dark_pixels])

    slopes_blanked_outer[band] = np.array([])
    for bolo in d:
        pixelnum = int(bolo.split('.')[0].split('/')[1])
        pixel_xy = pxtools.pixelnum2XY([pixelnum])[0]
        r = np.sqrt(pixel_xy[0]**2 + pixel_xy[1]**2)
        
        if str(d[bolo]['pstring']).split('/')[0] == '0136' and \
           bolo.split('.')[1] == str(band) and r>20.:
            slopes_blanked_outer[band] = np.append(slopes_blanked_outer[band], d[bolo]['slope_CL'])
    median_slopes_blanked_outer[band] = np.median(slopes_blanked_outer[band])

    slopes_optical_corrected[band] = slopes_optical[band] - median_slopes_blanked_outer[band]
    efficiency_optical_corrected[band] = efficiency_optical[band] * \
                                         slopes_optical_corrected[band] / \
                                         slopes_optical[band]

plt.figure()
for jband, band in enumerate([90, 150, 220]):
    plt.hist(slopes_optical[band]*1e12, np.linspace(-0.75, 0, 41),
             label='{} GHz'.format(band), histtype='stepfilled', alpha=0.3, color='C{}'.format(jband))
    plt.hist(slopes_optical[band]*1e12, np.linspace(-0.75, 0, 41),
             histtype='step', color='C{}'.format(jband))
plt.xlim([-0.75, 0])
plt.legend()
plt.title('With UC stage temp. correction')
plt.xlabel('slope [pW/K]')
plt.tight_layout()
plt.savefig('optical_slopes.png', dpi=200)

plt.figure()
for jband, band in enumerate([90, 150, 220]):
    plt.hist(slopes_optical_corrected[band]*1e12, np.linspace(-0.75, 0, 41),
             label='{} GHz'.format(band), histtype='stepfilled', alpha=0.3, color='C{}'.format(jband))
    plt.hist(slopes_optical_corrected[band]*1e12, np.linspace(-0.75, 0, 41),
             histtype='step', color='C{}'.format(jband))
plt.xlim([-0.75, 0])
plt.legend()
plt.title('With UC stage temp. correction & dark bolo correction')
plt.xlabel('slope [pW/K]')
plt.tight_layout()
plt.savefig('optical_slopes_corrected.png', dpi=200)

plt.figure()
for jband, band in enumerate([90, 150, 220]):
    plt.hist(efficiency_optical[band], np.linspace(0, 1.2, 41),
             label='{} GHz'.format(band), histtype='stepfilled', alpha=0.3, color='C{}'.format(jband))
    plt.hist(efficiency_optical[band], np.linspace(0, 1.2, 41),
             histtype='step', color='C{}'.format(jband))
plt.legend()
plt.xlim([0, 1.2])
plt.title('With UC stage temp. correction')
plt.xlabel('optical efficiency')
plt.tight_layout()
plt.savefig('optical_efficiency.png', dpi=200)

plt.figure()
for jband, band in enumerate([90, 150, 220]):
    plt.hist(efficiency_optical_corrected[band], np.linspace(0, 1.2, 41),
             label='{} GHz'.format(band), histtype='stepfilled', alpha=0.3, color='C{}'.format(jband))
    plt.hist(efficiency_optical_corrected[band], np.linspace(0, 1.2, 41),
             histtype='step', color='C{}'.format(jband))
plt.legend()
plt.xlim([0, 1.2])
plt.title('With UC stage temp. correction & dark bolo correction')
plt.xlabel('optical efficiency')
plt.tight_layout()
plt.savefig('optical_efficiency_corrected.png', dpi=200)


plt.figure()
for jband, band in enumerate([90, 150, 220]):
    plt.hist(slopes_blanked[band]*1e12, np.linspace(-0.5, 0.2, 41),
             label='{} GHz'.format(band), histtype='stepfilled', alpha=0.3, color='C{}'.format(jband))
    plt.hist(slopes_blanked[band]*1e12, np.linspace(-0.5, 0.2, 41),
             histtype='step', color='C{}'.format(jband))

    # plt.hist(slopes_blanked[band]*1e12, np.linspace(-0.5, 0.2, 41),
    #          label='{} GHz'.format(band), histtype='step')
plt.legend()
plt.xlim([-0.5, 0.2])
plt.xlabel('slope [pW/K]')
plt.tight_layout()
plt.savefig('blanked_slopes.png', dpi=200)

plt.figure()
for jband, band in enumerate([90, 150, 220]):
    plt.hist(slopes_blanked_outer[band]*1e12, np.linspace(-0.5, 0.2, 41),
             label='{} GHz'.format(band), histtype='stepfilled', alpha=0.3, color='C{}'.format(jband))
    plt.hist(slopes_blanked_outer[band]*1e12, np.linspace(-0.5, 0.2, 41),
             histtype='step', color='C{}'.format(jband))

    # plt.hist(slopes_blanked_outer[band]*1e12, np.linspace(-0.5, 0.5, 41),
    #          label='{} GHz'.format(band), histtype='step')
plt.legend()
plt.title('blanked off detectors with r>2cm\n'
          'median slope for 90 / 150 / 220: {:.3f} / {:.3f} / {:.3f}'.format(median_slopes_blanked_outer[90]*1e12,
                                                            median_slopes_blanked_outer[150]*1e12,
                                                            median_slopes_blanked_outer[220]*1e12))
plt.xlim([-0.5, 0.2])
plt.xlabel('slope [pW/K]')
plt.tight_layout()
plt.savefig('blanked_outer_slopes.png', dpi=200)



plt.figure()
for band in slopes_dark:
    plt.hist(slopes_dark[band]*1e12, np.linspace(-0.5, 0.5, 41),
             label='{} GHz'.format(band), histtype='step')
plt.legend()
plt.savefig('dark_slopes.png', dpi=200)



plt.figure()
for band in [90, 150, 220]:
    plt.plot(freqs_optical[band]*1e-6, efficiency_optical[band],
             'o', label='{} GHz'.format(band))
plt.xlabel('bias frequency [MHz]')
plt.ylabel('uncorrected optical efficiency')
plt.tight_layout()
plt.savefig('optical_effVfreq.png', dpi=200)
