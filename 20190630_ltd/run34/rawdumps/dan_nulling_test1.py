import pickle
import matplotlib.pyplot as plt
import numpy as np
#import pydfmux

# hwm_file = '/home/adama/hardware_maps/fnal/run34/slots12/hwm.yaml'
# hwm = pydfmux.load_session(open(hwm_file, 'r'))['hardware_map']
# rchans = hwm.readout_channels_from_pstring('0135/2/1/{1:10}')
# bias_frequencies = [rc.lc_channel.frequency for rc in rchans]
fname1 = ['/daq/pydfmux_output/20181011/20181012_005330_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_3_OUTPUT.pkl',
          '/daq/pydfmux_output/20181011/20181012_005752_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_3_OUTPUT.pkl']
fname2 = ['/daq/pydfmux_output/20181011/20181012_005330_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_4_OUTPUT.pkl',
          '/daq/pydfmux_output/20181011/20181012_005752_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_4_OUTPUT.pkl']

for jdump in [0,1]:
    with open(fname1[jdump], 'r') as f:
        d1 = pickle.load(f) #, encoding='latin1')

    with open(fname2[jdump], 'r') as f:
        d2 = pickle.load(f) #, encoding='latin1')

    plt.figure()
    freq1 = d1['full_data']['spectrum_freqs']
    plt.plot(freq1[freq1>1000], d1['full_data']['spectrum_real'][freq1>1000], linewidth=1)
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('SQUID output [uV/rtHz]')
    plt.xlim([1000, 10e6])
    plt.tight_layout()
    plt.savefig('fnal_run34_dannulling_test1_rawdump_0135_2_3_dump{}.png'.format(jdump), dpi=200)
    plt.close()

    plt.figure()
    freq2 = d2['full_data']['spectrum_freqs']
    plt.plot(freq2[freq2>1000], d2['full_data']['spectrum_real'][freq2>1000], linewidth=1)
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('SQUID output [uV/rtHz]')
    plt.xlim([1000, 10e6])
    plt.tight_layout()
    plt.savefig('fnal_run34_dannulling_test1_rawdump_0135_2_4_dump{}.png'.format(jdump), dpi=200)
    plt.close()
