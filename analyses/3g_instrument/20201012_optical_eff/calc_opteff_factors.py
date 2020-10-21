from spt3g import core, calibration, dfmux
import numpy as np
from spt3g.dfmux import HousekeepingForBolo
from spt3g.dfmux.unittransforms import bolo_bias_voltage_rms
import matplotlib.pyplot as plt

wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180',
          'w181', 'w188', 'w203', 'w204', 'w206']

kcmb_conversion_factors = {
  # Best guess for W28A2 based on a map using RCW28 calibration
  'W28A2': {
    90.0*core.G3Units.GHz: 4.858e-8*core.G3Units.K,
    150.0*core.G3Units.GHz: 3.536e-8*core.G3Units.K,
    220.0*core.G3Units.GHz: 6.560e-8*core.G3Units.K,
  },
  'RCW38': {
    90.0*core.G3Units.GHz: 4.0549662e-07*core.G3Units.K,
    150.0*core.G3Units.GHz: 2.5601153e-07*core.G3Units.K,
    220.0*core.G3Units.GHz: 2.8025804e-07*core.G3Units.K,
  },
  'MAT5A': {
    90.0*core.G3Units.GHz: 2.5738063e-07*core.G3Units.K, # center (608, 555)
    150.0*core.G3Units.GHz: 1.7319235e-07*core.G3Units.K,
    220.0*core.G3Units.GHz: 2.145164e-07*core.G3Units.K,
  },
}


# files to use
calfile = '/sptgrid/analysis/calibration/calframe/ra0hdec-52.25/86008954.g3'
biasfile = '/spt/data/bolodata/fullrate/noise/81433244/0000.g3'


# get data
dcal = list(core.G3File(calfile))[0]
d = list(core.G3File(biasfile))


# get dfmux efficiency
bolos = []
wafers = []
bands = []
eff_RCW38 = []
pWperK_RCW38 = []
pWperKperVb_RCW38 = []
for bolo in dcal['BolometerProperties'].keys():
    if bolo in dcal['CalibratorResponse'].keys() and \
       bolo in dcal['RCW38FluxCalibration'].keys() and \
       bolo in dcal['RCW38IntegralFlux'].keys():
        bolos.append(bolo)
        wafers.append(dcal['BolometerProperties'][bolo].wafer_id)
        bands.append(dcal['BolometerProperties'][bolo].band)
        eff_RCW38.append(-1 * dcal['CalibratorResponse'][bolo] * \
                            dcal['RCW38FluxCalibration'][bolo] * \
                            dcal['RCW38IntegralFlux'][bolo] / \
                            kcmb_conversion_factors['RCW38'][bands[-1]] / \
                        (1e-12*core.G3Units.watt / core.G3Units.kelvin))
        pWperK_RCW38.append(-1 * dcal['CalibratorResponse'][bolo] * \
                            dcal['RCW38FluxCalibration'][bolo] * \
                            dcal['RCW38IntegralFlux'][bolo] / \
                        (1e-12*core.G3Units.watt / core.G3Units.kelvin))
bolos = np.array(bolos)
wafers = np.array(wafers)
bands = np.array(bands)
eff_RCW38 = np.array(eff_RCW38)
pWperK_RCW38 = np.array(pWperK_RCW38)


# get bias frequencies
bias_freq = []
bias_voltage_rms = []
for bolo in bolos:
    # try:
    chan = HousekeepingForBolo(d[3]["DfMuxHousekeeping"], d[2]['WiringMap'], bolo)
    bias_freq.append(chan.carrier_frequency / core.G3Units.Hz)
    bias_voltage_rms.append(bolo_bias_voltage_rms(d[2]['WiringMap'],
                                                d[3]["DfMuxHousekeeping"],
                                                bolo,
                                                d[2]['ReadoutSystem'],
                                                tf='spt3g_filtering_2017_full') / core.G3Units.V)
    # except:
    #     pass
bias_freq = np.array(bias_freq)
bias_voltage_rms = np.array(bias_voltage_rms)

effperV = eff_RCW38 / bias_voltage_rms


# calculate rescalings
pWperKperV_rescaled = []
pWperK_rescaled = []
for wafer in wafer_list:
    scalings = {90: 1,
                150: np.median(effperV[(bands==90*core.G3Units.GHz) & \
                                         (wafers==wafer) & \
                                         (bias_freq>2.3e6) & \
                                         np.isfinite(effperV)]) / \
                     np.median(effperV[(bands==150*core.G3Units.GHz) & \
                                         (wafers==wafer) & \
                                         (bias_freq<2.7e6) & \
                                         np.isfinite(effperV)]),
                220: np.median(effperV[(bands==150*core.G3Units.GHz) & \
                                         (wafers==wafer) & \
                                         (bias_freq>3.4e6) & \
                                         np.isfinite(effperV)]) / \
                     np.median(effperV[(bands==220*core.G3Units.GHz) & \
                                         (wafers==wafer) & \
                                         (bias_freq<3.8e6) & \
                                         np.isfinite(effperV)])}
    scalings[220] = scalings[150] * scalings[220]


# make diagnostic plots
f_plot = [bias_freq[bolo] for bolo in bias_freq if bolo in pWperK_RCW38]
pWperK_plot = [pWperK_RCW38[bolo] for bolo in bias_freq if bolo in pWperK_RCW38]
eff_plot = [eff_RCW38[bolo] for bolo in bias_freq if bolo in pWperK_RCW38]
plt.plot(bias_freq, eff_RCW38, 'o', markersize=2)
plt.xlim([1.2e6, 5.5e6])

plt.show()