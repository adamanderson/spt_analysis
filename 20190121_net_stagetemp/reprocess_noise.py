import os

noise_script_path = '/home/adama/SPT/spt3g_software/calibration/scripts/analyze_noise.py'
noise_cal_dir = '/home/adama/SPT/spt_analysis/20190121_net_stagetemp/calframe_corrected/'
noise_raw_dir = '/spt/data/bolodata/fullrate/noise/'
noise_obsids = [63084862, 63098693, 63218920, 63227942,
                63227942, 63380372, 63640406, 63650590,
                63661173, 63689042, 63728180,
                64576620, 64591411, 64606397, 64685912, 64685310]
no_opacity_correction = [True, True, True, True,
                         True, True, True, True,
                         True, True, True,
                         False, False, False, False, False]

for jobs, obsid in enumerate(noise_obsids):
    noise_cal_path = '{}/{}_calframe_corrected.g3'.format(noise_cal_dir, obsid)
    noise_raw_path = '{}/{}/0000.g3'.format(noise_raw_dir, obsid)
    processed_noise_path = 'noise_corrected/{}_processed_noise.g3'.format(obsid)

    print('Processing {}'.format(obsid))
    if os.path.exists(noise_cal_path) and \
       os.path.exists(noise_raw_path):
        if no_opacity_correction[jobs]:
            opacitystr = '--no-opacity-correction'
        else:
            opacitystr = ''
        os.system('python {} {} {} -o {} '
                  '--cal-sn-threshold 10 {}'.format(noise_script_path,
                                                    noise_cal_path,
                                                    noise_raw_path,
                                                    processed_noise_path,
                                                    opacitystr))
    else:
        print('Missing input files. Skipping.')
