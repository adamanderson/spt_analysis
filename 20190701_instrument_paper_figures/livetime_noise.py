import numpy as np
import matplotlib.pyplot as plt
from spt3g import core, calibration
import os.path
from datetime import datetime
from spt3g.std_processing import obsid_to_g3time
import matplotlib
import matplotlib.dates as mdates
from glob import glob
from spt3g.mapspectra.apodmask import make_border_apodization

color = {90:'C0', 150:'C1', 220:'C2'}
fields = ['ra0hdec-44.75', 'ra0hdec-52.25', 'ra0hdec-59.75', 'ra0hdec-67.25']
tstart = datetime(year=2019, month=1, day=1).timestamp()
tstop = datetime(year=2019, month=9, day=10).timestamp()
sunset = mdates.date2num(datetime(year=2019, month=3, day=21))


# Let's calculate the duration of a field scan excluding turnarounds
# (i.e the relevant length for noise estimates)
scan_time = 0
rawdata_path = '/spt/data/bolodata/downsampled/ra0hdec-44.75/84783979'
d = list(core.G3File(os.path.join(rawdata_path, '0000.g3')))
for fname in glob(os.path.join(rawdata_path, '0*.g3')):
    print(fname)
    d = list(core.G3File(os.path.join(fname)))
    for fr in d:
        if fr.type == core.G3FrameType.Scan and \
            'Turnaround' not in fr:
            scan_time += (fr['DetectorSampleTimes'][-1].time -
                          fr['DetectorSampleTimes'][0].time)

# load the coadded maps to get the weights and area of the uniform coverage region
coadd_path = '/spt/user/weiquan/map_quality/hi_res_maps/yearly/2019/'
coadd_fnames = {band: {field: os.path.join(coadd_path, 'coadded_maps_from_{}_{}GHz.g3.gz'.format(field, band))
                       for field in fields} for band in [90, 150, 220]}
coadd_TT_weights = {}
weights_ratio = {}
area_uniform_coverage = {}
for band in coadd_fnames.keys():
    if band not in coadd_TT_weights.keys():
        coadd_TT_weights[band] = {}
        weights_ratio[band] = {}
        area_uniform_coverage[band] = {}
        
    for field in fields:
        print('Loading: {}'.format(coadd_fnames[band][field]))
        coadd_data = list(core.G3File(coadd_fnames[band][field]))

        # get weights
        coadd_TT_weights[band][field] = coadd_data[2]['Wunpol'].TT + coadd_data[3]['Wunpol'].TT

        # calculate apodization mask
        apod_mask_uniform = make_border_apodization(coadd_TT_weights[band][field],
                                                    apod_type='tophat',
                                                    weight_threshold=0.9)

        weights_ratio[band][field] = np.sum(coadd_TT_weights[band][field] * apod_mask_uniform) / \
                                np.sum(coadd_TT_weights[band][field])
        area_uniform_coverage[band][field] = np.sum(apod_mask_uniform) * (apod_mask_uniform.res)**2

# load the analysis summary files to get the noise levels
noise_summary_path = '/spt/user/weiquan/map_quality/hi_res_maps/monthly/all_months'
noise_summary_fnames = {band: os.path.join(noise_summary_path, 'all_analysis_results_{}GHz.g3'.format(band))
                                          for band in [90, 150, 220]}
obsids_nets = {}
noises = {}
nets = {}
for band in noise_summary_fnames.keys():
    if band not in obsids_nets.keys():
        obsids_nets[band] = {}
        noises[band] = {}
        nets[band] = {}
        
    noise_summary = list(core.G3File(noise_summary_fnames[band]))
        
    for field in fields:

        obsids_nets[band][field] = np.array([k for k in noise_summary[0]["NoiseLevelsFromIndividualTMaps"]\
                                                                     [field].keys()])
        noises[band][field] = np.array([noise_summary[0]["NoiseLevelsFromIndividualTMaps"][field][k] \
                           for k in noise_summary[0]["NoiseLevelsFromIndividualTMaps"][field].keys()])
        nets[band][field] = noises[band][field] / np.sqrt(area_uniform_coverage[band][field]) * \
                        np.sqrt(scan_time * weights_ratio[band][field])

# find all observations with obsids that fall between start and stop
obsids_all = []
dirnames_all = []
for source in ['ra0hdec-44.75', 'ra0hdec-52.25',
               'ra0hdec-59.75', 'ra0hdec-67.25']:
    dirnames = np.array(glob('/spt/data/bolodata/downsampled/{}/*'.format(source)))
    obsids = np.array([int(dirname.split('/')[-1]) for dirname in dirnames])
    times = np.array([obsid_to_g3time(obsid).time/core.G3Units.second
                      for obsid in obsids])
    obsids_all = np.append(obsids_all, obsids[(times < tstop) & (times > tstart)])
    dirnames_all = np.append(dirnames_all, dirnames[(times < tstop) & (times > tstart)])

# find start and stop time of each observation
obs_tstart = []
obs_tstop = []
for dirname in dirnames_all:
    print(dirname)
    f = core.G3File('{}/0000.g3'.format(dirname))
    fr = f.next()
    try:
        obs_tstart.append(fr["ObservationStart"].time/core.G3Units.second)
        obs_tstop.append(fr["ObservationStop"].time/core.G3Units.second)
    except KeyError:
        pass
obs_tstart = np.sort(np.array(obs_tstart))
obs_tstop = np.sort(np.array(obs_tstop))
tlive = obs_tstop - obs_tstart
epoch_tstart = mdates.epoch2num(obs_tstart)

# set up months boundaries for x ticks
month_times = np.array([mdates.date2num(datetime(year=2019, month=jmonth, day=1)) for jmonth in np.arange(1,10)])
month_times = np.append(month_times, mdates.date2num(datetime(year=2019, month=9, day=10)))

# set up bin edges for binning
bin_times = np.linspace(mdates.epoch2num(tstart),
                        mdates.epoch2num(tstop), 30) 
month_lengths = bin_times[1:] - bin_times[:-1]
jbins = np.digitize(mdates.epoch2num(obs_tstart), bin_times) - 1


matplotlib.rcParams.update({'font.size': 13})

fig = plt.figure(figsize=(12,7))

ax_top = plt.subplot(3,1,1)
plt.ylabel('livetime fraction\non CMB')
_ = plt.hist(mdates.epoch2num(obs_tstart),
             weights=tlive / (3600*24) / month_lengths[jbins],
             bins=bin_times,
             color='C1', alpha=0.3)
_ = plt.hist(mdates.epoch2num(obs_tstart),
             weights=tlive / (3600*24) / month_lengths[jbins],
             bins=bin_times,
             color='C1', histtype='step')
plt.ylim([0,0.8])
plt.grid()


plt.subplot(3,1,2,sharex=ax_top)
for band in nets.keys():
    for field in fields:
        times_nets = np.array([mdates.epoch2num(obsid_to_g3time(ob).time/core.G3Units.sec) \
                               for ob in obsids_nets[band][field]])
        net_threshold = 6*core.G3Units.microkelvin * np.sqrt(core.G3Units.sec)
        plt.plot(times_nets[nets[band][field] > net_threshold],
                 nets[band][field][nets[band][field] > net_threshold] / \
                 (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec)), '.',
                 markersize=2, color=color[band])
plt.ylim([0, 39.7])
plt.xlabel('observation ID')
plt.ylabel('NET [$\mu$K$\sqrt{s}$]')
plt.tight_layout()
plt.grid()


band_labels = {90:'95 GHz', 150:'150 GHz', 220:'220 GHz'}
plt.subplot(3,1,3,sharex=ax_top)
for band in [90, 150, 220]:
    noise_dict = {}
    obsids_dict = {}
    for field in fields:
        d = list(core.G3File('/home/nadolski/3G_INST_PAPER_FIG_GEN/data/'
                             'some_analysis_results_{}GHz.g3'.format(band)))[0]
        noise = np.array([d["NoiseLevelsFromCoaddedTMaps"][field][obsid] \
                          for obsid in d["NoiseLevelsFromCoaddedTMaps"][field].keys()])
        noise = noise / (core.G3Units.microkelvin * core.G3Units.arcmin)
        obsids = [int(obsid) for obsid in d["NoiseLevelsFromCoaddedTMaps"][field].keys()]
        noise_dict[field] = noise
        obsids_dict[field] = obsids
        
    max_min_obsid = np.max([np.min(obsids_dict[field]) for field in obsids_dict])
    min_max_obsid = np.min([np.max(obsids_dict[field]) for field in obsids_dict])
    obsids_interp = np.linspace(max_min_obsid, min_max_obsid, 500)
    times_interp = np.array([mdates.epoch2num(obsid_to_g3time(ob).time/core.G3Units.sec) for ob in obsids_interp])
    noise_interp = np.sum(np.vstack([np.interp(obsids_interp, obsids_dict[field], noise_dict[field])
                           for field in fields]), axis=0) / 4
    
    plt.semilogy(times_interp, noise_interp, label=band_labels[band])


plt.legend()
plt.grid()
plt.ylabel('cumulative map\ndepth [$\mu$K arcmin]')

plt.tight_layout()
fig.subplots_adjust(hspace=0)

plt.savefig('livetime_noise_summary.pdf')
