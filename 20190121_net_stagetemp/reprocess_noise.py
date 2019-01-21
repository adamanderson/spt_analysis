import os

noise_script_path = '/home/adama/SPT/spt3g_software/calibration/scripts/analyze_noise.py'
noise_cal_dir = '/home/adama/SPT/spt_analysis/20190121_net_stagetemp/calframe_corrected/'
noise_raw_dir = '/spt/data/bolodata/fullrate/noise/'
noise_obsids = [64576620, 64591411, 64606397, 64685912, 64685310]

for obsid in noise_obsids:
    noise_cal_path = '{}/{}_calframe_corrected.g3'.format(noise_cal_dir, obsid)
    noise_raw_path = '{}/{}/0000.g3'.format(noise_raw_dir, obsid)
    processed_noise_path = 'noise_corrected/{}_processed_noise.g3'.format(obsid)

    print('Processing {}'.format(obsid))
    if os.path.exists(noise_cal_path) and \
       os.path.exists(noise_raw_path):
        print('python {} {} {} -o {} '
                  '--cal-sn-threshold 10'.format(noise_script_path,
                                                 noise_cal_path,
                                                 noise_raw_path,
                                                 processed_noise_path))
        os.system('python {} {} {} -o {} '
                  '--cal-sn-threshold 10'.format(noise_script_path,
                                                 noise_cal_path,
                                                 noise_raw_path,
                                                 processed_noise_path))
    else:
        print('Missing input files. Skipping.')
