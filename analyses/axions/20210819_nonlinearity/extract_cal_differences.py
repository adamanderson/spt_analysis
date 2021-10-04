import numpy as np
from spt3g import core, calibration, autoprocessing
import os.path
import pickle
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from scipy.signal import lombscargle

extract_data = False
plot = True
results_dict = {90: {'delta_cal_response':{},
                     'cal_response_before':{}},
                150:{'delta_cal_response':{},
                     'cal_response_before':{}}}

if extract_data:
    cal_path = '/sptgrid/analysis/calibration/calibrator'
    data_path = '/sptlocal/user/kferguson'
    boloprops = list(core.G3File('/sptgrid/analysis/calibration/boloproperties/60000000.g3'))[0]
    angle_data = {}
    for jband, band in enumerate([90,150]):
        with open(os.path.join(data_path, 'collated_final_angles_more_realizations_' + '%sGHz.pkl'%band),'rb') as f:
            angle_data[band] = pickle.load(f)

    scanify = autoprocessing.ScanifyDatabase(read_only=True)
    subfields = {}

    cal_obsids = np.sort([int(os.path.splitext(os.path.basename(path))[0]) \
                        for path in glob(os.path.join(cal_path, '*.g3'))])
    field_obsids = np.sort(list(angle_data[90].keys()))
    cal_obsids_before = []
    cal_obsids_after = []
    for obsid in field_obsids:
        cal_obsids_after.append(np.min(cal_obsids[cal_obsids>obsid]))
        cal_obsids_before.append(np.max(cal_obsids[cal_obsids<obsid]))
        subfields[obsid] = scanify.get_entries('.*', obsid, match_regex=True)[0][0]

    cal_response_diff_median = {90: {}, 150:{}}
    for jobs in range(len(field_obsids)):
        print(field_obsids[jobs])
        cal_data_after = list(core.G3File(os.path.join(cal_path, '{}.g3'.format(cal_obsids_after[jobs]))))
        cal_data_after_bolos = list(cal_data_after[0]["CalibratorResponse"].keys())
        cal_data_before = list(core.G3File(os.path.join(cal_path, '{}.g3'.format(cal_obsids_before[jobs]))))
        cal_data_before_bolos = list(cal_data_before[0]["CalibratorResponse"].keys())

        cal_response_difference = {90:[], 150:[]}
        cal_response_before = {90:[], 150:[]}
        for bolo in cal_data_before_bolos:
            try:
                band = boloprops["BolometerProperties"][bolo].band / core.G3Units.GHz
                cal_response_before[band].append(cal_data_before[0]["CalibratorResponse"][bolo])
                cal_response_difference[band].append(cal_data_after[0]["CalibratorResponse"][bolo] - cal_data_before[0]["CalibratorResponse"][bolo])
            except:
                pass

        for band in [90, 150]:
            cal_response_diff_arr = np.array(cal_response_difference[band])
            cal_response_before_arr = np.array(cal_response_before[band])
            results_dict[band]['delta_cal_response'][field_obsids[jobs]] = np.median(cal_response_diff_arr[np.isfinite(cal_response_diff_arr) & (cal_response_diff_arr!=0)])
            results_dict[band]['cal_response_before'][field_obsids[jobs]] = np.median(cal_response_before_arr[np.isfinite(cal_response_before_arr) & (cal_response_before_arr!=0)])

    with open('cal_response_diff.pkl', 'wb') as f:
        pickle.dump(results_dict, f)


if plot:
    data_path = '/sptlocal/user/kferguson'
    boloprops = list(core.G3File('/sptgrid/analysis/calibration/boloproperties/60000000.g3'))[0]
    angle_data = {}
    for jband, band in enumerate([90,150]):
        with open(os.path.join(data_path, 'collated_final_angles_more_realizations_' + '%sGHz.pkl'%band),'rb') as f:
            angle_data[band] = pickle.load(f)
    field_obsids = np.sort(list(angle_data[90].keys()))

    scanify = autoprocessing.ScanifyDatabase(read_only=True)
    subfields = {}
    for obsid in field_obsids:
        subfields[obsid] = scanify.get_entries('.*', obsid, match_regex=True)[0][0]
    subfield_names = np.unique(list(subfields.values()))

    with open('cal_response_diff.pkl', 'rb') as f:
        results_dict = pickle.load(f)
    
    for band in results_dict.keys():
        obsids_arr = np.array(list(subfields.keys()))
        subfields_arr = np.array(list(subfields.values()))
        delta_cal_response_arr = np.array([results_dict[band]['delta_cal_response'][obsid] for obsid in obsids_arr])
        cal_response_before_arr = np.array([results_dict[band]['cal_response_before'][obsid] for obsid in obsids_arr])
        angle_data_arr = np.array([angle_data[band][obsid]['angle'] / angle_data[band][obsid]['unc'] \
                                   for obsid in obsids_arr])
        for subfield in subfield_names:
            plt.figure(1)
            plt.plot(angle_data_arr[subfields_arr==subfield],
                     delta_cal_response_arr[subfields_arr==subfield] / (core.G3Units.watt*1e-15),
                     '.')
            plt.ylim([-0.1, 0.4])
            plt.xlabel('angle / uncertainty')
            plt.ylabel('delta responsivity [fW]')
            plt.title('{} GHz {}'.format(band, subfield))
            plt.savefig('delta_cal_vs_angle_{}_{}.png'.format(band, subfield), dpi=200)
            plt.close()

            plt.figure()
            plt.plot(angle_data_arr[subfields_arr==subfield],
                     delta_cal_response_arr[subfields_arr==subfield] / cal_response_before_arr[subfields_arr==subfield],
                     '.')
            plt.xlabel('angle / uncertainty')
            plt.ylabel('delta responsivity (fractional)')
            plt.title('{} GHz {}'.format(band, subfield))
            plt.savefig('delta_cal_frac_vs_angle_{}_{}.png'.format(band, subfield), dpi=200)
            plt.close()

            plt.figure()
            angles_this_subfield = angle_data_arr[subfields_arr==subfield]
            frac_cal_this_subfield = delta_cal_response_arr[subfields_arr==subfield] / cal_response_before_arr[subfields_arr==subfield]
            median_frac_cal = np.median(frac_cal_this_subfield)
            ksstat, pval = ks_2samp(angles_this_subfield[frac_cal_this_subfield < median_frac_cal],
                                    angles_this_subfield[frac_cal_this_subfield > median_frac_cal])
            plt.hist(angles_this_subfield[frac_cal_this_subfield < median_frac_cal],
                     bins=np.linspace(-5, 5, 26),
                     label='cal change < median', density=True, histtype='step')
            plt.hist(angles_this_subfield[frac_cal_this_subfield > median_frac_cal],
                     bins=np.linspace(-5, 5, 26),
                     label='cal change > median', density=True, histtype='step')
            plt.xlabel('angle / uncertainty')
            plt.title('{} GHz {}; KS p-value = {:.4E}'.format(band, subfield, pval))
            plt.legend()
            plt.savefig('angles_splitbycal_{}_{}.png'.format(band, subfield), dpi=200)
            plt.close()

            plt.figure()
            ls_freq = np.linspace(0.0005, 2, 4000)
            pgram = lombscargle(obsids_arr[subfields_arr==subfield] / (24*3600),
                                delta_cal_response_arr[subfields_arr==subfield] - \
                                    np.mean(delta_cal_response_arr[subfields_arr==subfield]), ls_freq)
            plt.plot(ls_freq, pgram)
            plt.xlabel('frequency [1/d]')
            plt.ylabel('Lomb-Scargle periodogram')
            plt.title('{}, {} GHz'.format(subfield, band))
            plt.tight_layout()
            plt.savefig('delta_cal_frac_lsgram_{}_{}GHz.png'.format(subfield, band), dpi=200)
            plt.close()

