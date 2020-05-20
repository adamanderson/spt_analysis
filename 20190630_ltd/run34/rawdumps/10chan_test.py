import pickle
import matplotlib.pyplot as plt
import numpy as np
import pydfmux

hwm_file = '/home/adama/hardware_maps/fnal/run34/slots12/hwm.yaml'
hwm = pydfmux.load_session(open(hwm_file, 'r'))['hardware_map']
rchans = hwm.readout_channels_from_pstring('0135/2/1/{1:10}')
bias_frequencies = [rc.lc_channel.frequency for rc in rchans]
fname1 = ['/daq/pydfmux_output/20181011/20181011_224733_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_1_OUTPUT.pkl',
          '/daq/pydfmux_output/20181011/20181011_232656_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_1_OUTPUT.pkl']
fname2 = ['/daq/pydfmux_output/20181011/20181011_224733_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_2_OUTPUT.pkl',
          '/daq/pydfmux_output/20181011/20181011_232656_take_rawdump/data/IceBoard_0135.Mezz_2.ReadoutModule_2_OUTPUT.pkl']

for jdump in [0,1]:
    with open(fname1[jdump], 'r') as f:
        d1 = pickle.load(f) #, encoding='latin1')

    with open(fname2[jdump], 'r') as f:
        d2 = pickle.load(f) #, encoding='latin1')

    fmin = 2*1.6e6
    fmax = 2*2.3e6

    plt.figure()
    freq1 = d1['full_data']['spectrum_freqs']
    plt.plot(freq1[(freq1>fmin) & (freq1<fmax)],
             d1['full_data']['spectrum_real'][(freq1>fmin) & (freq1<fmax)],
             linewidth=1)
    yl = plt.gca().get_ylim()
    for f in bias_frequencies:
        plt.plot([f, f], [0, 10], 'r--', linewidth=1)
    for f in bias_frequencies:
        plt.plot([2*f, 2*f], [0, 10], 'r--', linewidth=1)
    plt.plot(freq1[(freq1>fmin) & (freq1<fmax)],
             d1['full_data']['spectrum_real'][(freq1>fmin) & (freq1<fmax)],
             linewidth=1, color='C0')
    plt.xlim([fmin, fmax])
    plt.ylim(yl)
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('SQUID output [uV/rtHz]')
    plt.tight_layout()
    plt.savefig('fnal_run34_10chan_test_rawdump_0135_2_1_{}.png'.format(jdump), dpi=200)
    plt.close()
    
    plt.figure()
    freq2 = d2['full_data']['spectrum_freqs']
    plt.plot(freq2[(freq2>fmin) & (freq2<fmax)],
             d2['full_data']['spectrum_real'][(freq2>fmin) & (freq2<fmax)],
             linewidth=1)
    yl = plt.gca().get_ylim()
    for f in bias_frequencies:
        plt.plot([f, f], [0, 10], 'r--', linewidth=1)
    for f in bias_frequencies:
        plt.plot([2*f, 2*f], [0, 10], 'r--', linewidth=1)
    plt.plot(freq2[(freq2>fmin) & (freq2<fmax)],
             d2['full_data']['spectrum_real'][(freq2>fmin) & (freq2<fmax)],
             linewidth=1, color='C0')
    plt.xlim([fmin, fmax])
    plt.ylim(yl)
    plt.xlabel('bias frequency [Hz]')
    plt.ylabel('SQUID output [uV/rtHz]')
    plt.tight_layout()
    plt.savefig('fnal_run34_10chan_test_rawdump_0135_2_2_dump{}.png'.format(jdump), dpi=200)

    plt.close()
