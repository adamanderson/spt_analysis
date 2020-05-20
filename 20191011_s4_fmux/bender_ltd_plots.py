import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
import os

def make_hk_map(hkdata):
    mezz=[1,2]
    module=[1,2,3,4]

    boards=hkdata.keys()
    master_chlist=[]
    master_freq=[]
    master_amp=[]
    master_state=[]

    for kk in boards:
        for mz in mezz:
            for md in module:
                tmp= hkdata[kk].mezz[mz].modules[md]
                chlist=tmp.channels.keys()
                for ch in chlist:
                    sq=(mz-1)*4+md
                    master_chlist.append(hkdata[kk].serial.lstrip('0')+'/'+str(sq)+'/'+str(ch))
                    master_amp.append(tmp.channels[ch].carrier_amplitude)  ### note that this is in normalized units
                    master_freq.append(tmp.channels[ch].carrier_frequency/core.G3Units.MHz)   # in Mhz
                    master_state.append(tmp.channels[ch].state)

    master_chlist=np.array(master_chlist)
    master_freq=np.array(master_freq)
    master_amp=np.array(master_amp)
    master_state=np.array(master_state)

    return master_chlist,master_freq,master_amp,master_state

def get_dfmux_information(bolokey,wmap,master_chlist,master_freq,master_amp,master_state):
    #crate=wmap[bolokey].crate_serial
    #slot=wmap[bolokey].board_slot
    squid=wmap[bolokey].module
    channel=wmap[bolokey].channel
    serial=wmap[bolokey].board_serial
    search_str=str(serial)+'/'+str(squid+1)+'/'+str(channel+1)
    search_result=np.where(master_chlist==search_str)[0]
    if len(search_result)==1:
        bfreq=master_freq[search_result[0]]
        camp=master_amp[search_result[0]]
        state=master_state[search_result[0]]
    else:
        print('ambiguous search: '+bolokey+':'+search_str)
        bfreq=-1
        camp=-1
        state='-1'
    return bfreq,camp,state


# some useful noise functions
def readout_noise(x, readout):
    return np.sqrt(readout)*np.ones(len(x))
def photon_noise(x, photon, tau):
    return np.sqrt(photon / (1 + 2*np.pi*((x*tau)**2)))
def atm_noise(x, A, alpha):
    return np.sqrt(A * (x)**(-1*alpha))
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout + (A * (x)**(-1*alpha)) + photon / (1 + 2*np.pi*((x*tau)**2)))
def horizon_model(x, readout, A, alpha):
    return np.sqrt(readout + (A * (x)**(-1*alpha)))
def knee_func(x, readout, A, alpha, photon, tau):
    return (A * (x)**(-1*alpha)) - photon / (1 + 2*np.pi*((x*tau)**2)) - readout
def horizon_knee_func(x, readout, A, alpha):
    return (A * (x)**(-1*alpha)) - readout

horizondatafile='/home/adama/SPT/spt_analysis/20190329_gainmatching/horizon_noise_77863968_bender_ltd.g3'


fr = list(core.G3File(horizondatafile))[1]

band_numbers = {90.: 1, 150.: 2, 220.: 3}
subplot_numbers = {90.: 1, 150.: 1, 220.: 1}
band_labels = {90:'95', 150:'150', 220:'220'}

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,4))
fig.subplots_adjust(wspace=0)
for jband, band in enumerate([90., 150., 220.]):
    
    group = '{:.1f}_w180'.format(band)

    ff_diff = np.array(fr['AverageASDDiff']['frequency']/core.G3Units.Hz)
    ff_sum = np.array(fr['AverageASDSum']['frequency']/core.G3Units.Hz)
    asd_diff = np.array(fr['AverageASDDiff'][group]) / np.sqrt(2.)
    asd_sum = np.array(fr['AverageASDSum'][group]) / np.sqrt(2.)

    par = fr["AverageASDDiffFitParams"][group]
    ax[jband].loglog(ff_sum[ff_sum<75], asd_sum[ff_sum<75],
                     label='pair sum (measured)', color='0.6')
    ax[jband].loglog(ff_diff[ff_diff<75], asd_diff[ff_diff<75],
                     label='pair difference (measured)', color='k')
    ax[jband].loglog(ff_sum, atm_noise(ff_sum, par[1], par[2]) / np.sqrt(2.),
                     'C0--', label='low-frequency noise')
    ax[jband].loglog(ff_sum, readout_noise(ff_sum, par[0]) / np.sqrt(2.),
                     'C2--', label='white noise')
    ax[jband].loglog(ff_sum, horizon_model(ff_sum, *list(par)) / np.sqrt(2.),
                     'C3--', label='total noise model')

    ax[jband].set_title('{} GHz'.format(band_labels[band]))
    ax[jband].set_xlabel('frequency [Hz]')
    ax[jband].grid()
ax[0].set_ylabel('current noise [pA/$\sqrt{Hz}$]')

plt.ylim([5,1000])
plt.legend()
plt.tight_layout()
plt.savefig('w180_horizon_noise_ltd.pdf')

horizondatafile='/home/adama/SPT/spt_analysis/20190329_gainmatching/horizon_noise_77863968_bender_ltd_perbolo_only.g3'
fr = list(core.G3File(horizondatafile))[1]
bolokeys=fr['ASDFitParams'].keys()
param1=np.zeros(len(bolokeys))
param2=np.zeros(len(bolokeys))
param3=np.zeros(len(bolokeys))
onefknee=np.zeros(len(bolokeys))  # defined as where wnl = 1/f noise level in PSD (corner frequency)

for ii,kk in enumerate(bolokeys):
    if len(fr['ASDFitParams'][kk])>0:
        param1[ii]=fr['ASDFitParams'][kk][0]
        param2[ii]=fr['ASDFitParams'][kk][1]
        param3[ii]=fr['ASDFitParams'][kk][2]
        onefknee[ii]= (param2[ii]/param1[ii])**(1./param3[ii])
        # the model is ASD = np.sqrt(param1 + param2*frequency^(-1*param3))

wnl=np.sqrt(param1)

obsband=np.zeros(len(bolokeys))
calsn=np.zeros(len(bolokeys))
biasfreq=np.zeros(len(bolokeys))
biasamp=np.zeros(len(bolokeys))
bias_state=np.zeros(len(bolokeys),dtype='S10')
# load in the offline calibration file used in the analysis
calfile='/spt/data/bolodata/downsampled/noise/73798315/offline_calibration.g3'
caldata=list(core.G3File(calfile))[0]

fulldata=list(core.G3File(os.path.join(calfile.strip('offline_calibration.g3'),'0000.g3')))
for ff in fulldata:
    if ff.type==core.G3FrameType.Wiring:
        wmap=ff['WiringMap']
    if ff.type==core.G3FrameType.Scan:
        hkdata=ff['DfMuxHousekeeping']

master_chlist,master_freq,master_amp,master_state=make_hk_map(hkdata)
namb=0
gpsd=np.zeros(len(bolokeys))+1
for ii,kk in enumerate(bolokeys):
    obsband[ii]=caldata['BolometerProperties'][kk].band / core.G3Units.GHz
    calsn[ii]=caldata['CalibratorResponseSN'][kk]
    if np.all(np.abs(fr['ASD'][kk])<1e-10):
        gpsd[ii] =0
    bfreq,camp,state=get_dfmux_information(kk,wmap,master_chlist,master_freq,master_amp,master_state)
    biasfreq[ii]=bfreq
    biasamp[ii]=camp
    bias_state[ii]=state
    if bfreq==-1:
        namb+=1

calsn[np.isnan(calsn)] = 0.

# let's isolate the untuned etc
ginds=np.intersect1d(np.intersect1d(np.intersect1d(np.where(biasfreq>0),np.where(biasamp >0)),np.where(bias_state == b'tuned')),np.where(calsn >20.))

ginds=np.intersect1d(np.intersect1d(np.where(calsn>20.)[0],np.where(gpsd ==1)[0]),np.where(wnl > 1e-2)[0])
print(len(ginds))
print(len(biasfreq))

plt.figure()
plt.plot(biasfreq[ginds],wnl[ginds],'k.')
plt.xlabel('Bias Frequency (MHz)')
plt.ylabel('White Noise Level (pA/$\sqrt{Hz}$)')
plt.ylim(0,100)


plt.savefig('nei_vs_bfreq_horizon_77863968.png')

plt.figure()
plt.plot(biasfreq[ginds],onefknee[ginds],'k.')
plt.xlabel('Bias Frequency (MHz)')
plt.ylabel('1/f knee (Hz)')
plt.ylim(0,1)
plt.savefig('onefknee_vs_bfreq_horizon_77863968.png')

plt.figure()
plt.plot(biasfreq[ginds],np.sqrt(param2[ginds]),'k.')
plt.xlabel('Bias Frequency (MHz)')
plt.ylabel('$\sqrt{1/f Amplitude}$')
plt.ylim(0,40)
plt.savefig('onefamp_vs_bfreq_horizon_77863968.png')

wnlbins=np.arange(0,35,1)
plt.figure()
for band in [90, 150, 220]:
    print(len(wnl))
    print(len(ginds))
    print(len(obsband))
    plt.hist(wnl[ginds & (obsband==band)],wnlbins,label='Measured', alpha=0.5)#,histtype='step',linewidth=2.)
plt.grid('on')
plt.yticks(np.arange(0,2000,500))
plt.ylim(0,1700)
ylm=plt.ylim()
plt.plot(np.array([1,1])*19., np.array([0,ylm[1]]),'--',label='150 GHz\n Photon Noise')
plt.legend()
#plt.legend((p1,p2),('Measured', '150 GHz Photon Noise'),loc='best')
plt.xlabel('White Noise (pA/$\sqrt{Hz}$)')
plt.ylabel('Number of Readout Channels')
plt.xlim(0,35)
plt.savefig('nei_hist_ltd_horizon_77863968.png')

# onefkneebins=np.arange(0,0.4,0.01)
# plt.figure()
# plt.hist(onefknee[ginds],onefkneebins)
# plt.grid('on')
# plt.xlim(0,0.4)
# plt.yticks(np.arange(0,3000,500))
# plt.xlabel('1/f Knee in PSD (Hz)')
# plt.ylabel('Number of Readout Channels')
# plt.savefig('onefknee_hist_ltd_horizon_77863968.png')






plt.show()
