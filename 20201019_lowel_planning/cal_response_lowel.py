import numpy as np
from glob import glob
import os.path
from spt3g import core, calibration
import matplotlib.pyplot as plt

fname_bps = '/sptgrid/data/bolodata/downsampled/ra5hdec-24.5/100005793/nominal_online_cal.g3'
bps = list(core.G3File(fname_bps))[0]["NominalBolometerProperties"]
bolos_in_bps = np.array(list(bps.keys()))
autoproc_dir = '/sptgrid/analysis/calibration/'
cal_dir = os.path.join(autoproc_dir, 'calibrator')
fields = ['ra5hdec-24.5', 'ra5hdec-31.5']
band_list = [90, 150, 220]
wafer_list = ['w172', 'w174', 'w176', 'w177', 'w180', 'w181', 'w188', 'w203', 'w204', 'w206', 'all']
band_colors = {90:'C0', 150:'C1', 220:'C2'}

for field in fields:
    fields_dir = os.path.join(autoproc_dir, '{}/pointing'.format(field))

    deltat_middle = -234
    deltat_bottom = -73
    deltat_top = 6264
    deltat_tol = 30

    frac_cal_all_bottom_middle_10 = {wafer: {band: [] for band in band_list} for wafer in wafer_list}
    frac_cal_all_bottom_top_10 = {wafer: {band: [] for band in band_list} for wafer in wafer_list}
    frac_cal_all_bottom_middle_20 = {wafer: {band: [] for band in band_list} for wafer in wafer_list}
    frac_cal_all_bottom_top_20 = {wafer: {band: [] for band in band_list} for wafer in wafer_list}

    # mangle observations
    obsids_field = [int(os.path.splitext(os.path.basename(p))[0]) \
                    for p in glob(os.path.join(fields_dir, '*.g3'))]
    obsids_field = np.sort(obsids_field)
    obsids_cal   = [int(os.path.splitext(os.path.basename(p))[0]) \
                    for p in glob(os.path.join(cal_dir, '*.g3'))]
    obsids_cal   = np.sort(obsids_cal)

    obsids_middle = {}
    obsids_top = {}
    obsids_bottom = {}
    for obsid in obsids_field:
        oi_middle = obsids_cal[(obsids_cal > obsid + deltat_middle - deltat_tol) & \
                            (obsids_cal < obsid + deltat_middle + deltat_tol)]
        oi_top    = obsids_cal[(obsids_cal > obsid + deltat_top - deltat_tol) & \
                            (obsids_cal < obsid + deltat_top + deltat_tol)]
        oi_bottom = obsids_cal[(obsids_cal > obsid + deltat_bottom - deltat_tol) & \
                            (obsids_cal < obsid + deltat_bottom + deltat_tol)]

        if len(oi_middle)>0 and len(oi_top)>0 and len(oi_bottom)>0:
            obsids_middle[obsid] = oi_middle[0]
            obsids_top[obsid]    = oi_top[0]
            obsids_bottom[obsid] = oi_bottom[0]
    
    for oi_field in obsids_middle: #list(obsids_middle.keys())[:3]:
        d_bottom = list(core.G3File(os.path.join(cal_dir, '{}.g3'.format(obsids_bottom[oi_field]))))[0]
        d_middle = list(core.G3File(os.path.join(cal_dir, '{}.g3'.format(obsids_middle[oi_field]))))[0]
        d_top    = list(core.G3File(os.path.join(cal_dir, '{}.g3'.format(obsids_top[oi_field]))))[0]

        bolos_to_plot = np.array([bolo for bolo in d_middle['CalibratorResponse'].keys() \
                                  if bolo in bolos_in_bps])
        
        bands = np.array([bps[bolo].band/core.G3Units.GHz for bolo in bolos_to_plot])
        wafers = np.array([bps[bolo].wafer_id for bolo in bolos_to_plot])
        cal_response_bottom = np.array([d_bottom['CalibratorResponse'][bolo] \
                                        for bolo in bolos_to_plot]) / \
                            (core.G3Units.watt * 1e-15)
        cal_response_middle = np.array([d_middle['CalibratorResponse'][bolo] \
                                        for bolo in bolos_to_plot]) / \
                            (core.G3Units.watt * 1e-15)
        cal_response_top = np.array([d_top['CalibratorResponse'][bolo] \
                                        for bolo in bolos_to_plot]) / \
                            (core.G3Units.watt * 1e-15)
        frac_cal_response_bottom_middle = (cal_response_middle - cal_response_bottom) / cal_response_bottom
        frac_cal_response_middle_top = (cal_response_top - cal_response_middle) / cal_response_middle
        frac_cal_response_bottom_top = (cal_response_top - cal_response_bottom) / cal_response_bottom
        
        

        for wafer in wafer_list:
            # middle - bottom
            plt.figure()
            for band in band_list:
                if wafer == 'all':
                    wafer_selection = np.array([True for x in wafers])
                else:
                    wafer_selection = wafers == wafer
                ntotal = len(frac_cal_response_bottom_middle[np.isfinite(frac_cal_response_bottom_middle) & \
                                                             (bands == band) & (wafer_selection)])
                nlessthanthreshold = len(frac_cal_response_bottom_middle[np.isfinite(frac_cal_response_bottom_middle) & \
                                                             (bands == band) & (wafer_selection) & \
                                                             (np.abs(frac_cal_response_bottom_middle)<0.2)])
                good_bolos_frac = nlessthanthreshold / ntotal
                frac_cal_all_bottom_middle_20[wafer][band].append(good_bolos_frac)
                nlessthanthreshold = len(frac_cal_response_bottom_middle[np.isfinite(frac_cal_response_bottom_middle) & \
                                                             (bands == band) & (wafer_selection) & \
                                                             (np.abs(frac_cal_response_bottom_middle)<0.1)])
                good_bolos_frac = nlessthanthreshold / ntotal
                frac_cal_all_bottom_middle_10[wafer][band].append(good_bolos_frac)
                plt.hist(frac_cal_response_bottom_middle[np.isfinite(frac_cal_response_bottom_middle) & \
                                                         (bands == band) & (wafer_selection)],
                        bins=np.linspace(-1,1,51), color=band_colors[band],
                        histtype='step', label='{} GHz (stable cal. bolos = {:.1f}%)'.format(band, 100*good_bolos_frac))
            plt.title('{}: {}'.format(field, oi_field))
            plt.legend()
            plt.xlabel('fractional change in cal. response')
            plt.tight_layout()
            plt.savefig('figures/mid_bottom_{}_{}_{}.png'.format(field, oi_field, wafer), dpi=200)
            plt.close()

            # top - middle
            plt.figure()
            for band in band_list:
                if wafer == 'all':
                    wafer_selection = np.array([True for x in wafers])
                else:
                    wafer_selection = wafers == wafer
                ntotal = len(frac_cal_response_middle_top[np.isfinite(frac_cal_response_middle_top) & \
                                                             (bands == band) & (wafer_selection)])
                nlessthanthreshold = len(frac_cal_response_middle_top[np.isfinite(frac_cal_response_middle_top) & \
                                                             (bands == band) & (wafer_selection) & \
                                                             (np.abs(frac_cal_response_middle_top)<0.1)])
                good_bolos_frac = nlessthanthreshold / ntotal
                plt.hist(frac_cal_response_middle_top[np.isfinite(frac_cal_response_middle_top) & \
                                                      (bands == band) & (wafer_selection)],
                        bins=np.linspace(-1,1,51), color=band_colors[band],
                        histtype='step', label='{} GHz (stable cal. bolos = {:.1f}%)'.format(band, 100*good_bolos_frac))
            plt.title('{}: {}'.format(field, oi_field))
            plt.legend()
            plt.xlabel('fractional change in cal. response')
            plt.tight_layout()
            plt.savefig('figures/top_mid_{}_{}_{}.png'.format(field, oi_field, wafer), dpi=200)
            plt.close()

            # top - bottom
            plt.figure()
            for band in band_list:
                if wafer == 'all':
                    wafer_selection = np.array([True for x in wafers])
                else:
                    wafer_selection = wafers == wafer
                ntotal = len(frac_cal_response_bottom_top[np.isfinite(frac_cal_response_bottom_top) & \
                                                             (bands == band) & (wafer_selection)])
                nlessthanthreshold = len(frac_cal_response_bottom_top[np.isfinite(frac_cal_response_bottom_top) & \
                                                             (bands == band) & (wafer_selection) & \
                                                             (np.abs(frac_cal_response_bottom_top)<0.2)])
                good_bolos_frac = nlessthanthreshold / ntotal
                frac_cal_all_bottom_top_20[wafer][band].append(good_bolos_frac)
                nlessthanthreshold = len(frac_cal_response_bottom_top[np.isfinite(frac_cal_response_bottom_top) & \
                                                             (bands == band) & (wafer_selection) & \
                                                             (np.abs(frac_cal_response_bottom_top)<0.1)])
                good_bolos_frac = nlessthanthreshold / ntotal
                frac_cal_all_bottom_top_10[wafer][band].append(good_bolos_frac)
                plt.hist(frac_cal_response_bottom_top[np.isfinite(frac_cal_response_bottom_top) & \
                                                      (bands == band) & (wafer_selection)],
                        bins=np.linspace(-1,1,51), color=band_colors[band],
                        histtype='step', label='{} GHz (stable cal. bolos = {:.1f}%)'.format(band, 100*good_bolos_frac))
            plt.title('{}: {}'.format(field, oi_field))
            plt.legend()
            plt.xlabel('fractional change in cal. response')
            plt.tight_layout()
            plt.savefig('figures/top_bottom_{}_{}_{}.png'.format(field, oi_field, wafer), dpi=200)
            plt.close()


    for wafer in wafer_list:
        for band in band_list:
            plt.figure()
            cal_frac_change = frac_cal_all_bottom_top_10[wafer][band]
            plt.plot(np.arange(len(cal_frac_change)), cal_frac_change,
                     color=band_colors[band], linestyle='-', marker='o',
                     label='{} GHz (top - bottom)'.format(band))
            cal_frac_change = frac_cal_all_bottom_middle_10[wafer][band]
            plt.plot(np.arange(len(cal_frac_change)), cal_frac_change,
                     color=band_colors[band], linestyle='--', marker='o',
                     label='{} GHz (middle - bottom)'.format(band))
            plt.legend()
            plt.title('{}, {}, {} GHz'.format(field, wafer, band))
            plt.xlabel('observation index')
            plt.ylabel('fraction of bolometer with $\Delta$(cal. response) < 10%')
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig('figures/cal_change_10p_{}_{}_{}.png'.format(field, wafer, band))
            plt.close()


            plt.figure()
            cal_frac_change = frac_cal_all_bottom_top_20[wafer][band]
            plt.plot(np.arange(len(cal_frac_change)), cal_frac_change,
                     color=band_colors[band], linestyle='-', marker='o',
                     label='{} GHz (top - bottom)'.format(band))
            cal_frac_change = frac_cal_all_bottom_middle_20[wafer][band]
            plt.plot(np.arange(len(cal_frac_change)), cal_frac_change,
                     color=band_colors[band], linestyle='--', marker='o',
                     label='{} GHz (middle - bottom)'.format(band))
            plt.legend()
            plt.title('{}, {}, {} GHz'.format(field, wafer, band))
            plt.xlabel('observation index')
            plt.ylabel('fraction of bolometer with $\Delta$(cal. response) < 20%')
            plt.ylim([0,1])
            plt.tight_layout()
            plt.savefig('figures/cal_change_20p_{}_{}_{}.png'.format(field, wafer, band))
            plt.close()
