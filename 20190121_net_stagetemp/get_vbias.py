# script for extracting voltage biases from noise stare data
from spt3g import core, dfmux, calibration
from spt3g.dfmux.unittransforms import bolo_bias_voltage_rms
import glob
import os.path
import pickle

rawdata_path = '/spt/data/bolodata/fullrate/noise/'
obsids = [63084862, 63098693, 63218920, 63227942,
          63305224, 63380372, 63640406, 63650590,
          63661173, 63689042, 63728180, 64576620,
          64591411, 64606397, 64685912, 64701072]

data = {}
for obsid in obsids:
    print('Processing {}'.format(obsid))
    
    data[obsid] = {}
    fname = os.path.join(rawdata_path, str(obsid), '0000.g3')
    fname_cal = os.path.join(rawdata_path, str(obsid), 'offline_calibration.g3')
    if os.path.exists(fname) and os.path.exists(fname_cal):
        d = [fr for fr in core.G3File(fname)]
        wm = d[1]['WiringMap']
        hk = d[2]["DfMuxHousekeeping"]
        bolos = d[2]['RawTimestreams_I'].keys()
        vbias = {bolo: bolo_bias_voltage_rms(wm, hk, bolo, 'ICE', 'spt3g_filtering_2017_full')
                 for bolo in bolos}
        data[obsid]['vbiasrms'] = vbias
        
        dcal = [fr for fr in core.G3File(fname_cal)]
        data[obsid]['boloprops'] = dcal[0]["BolometerProperties"]

with open('vbias.pkl', 'wb') as f:
    pickle.dump(data, f)
