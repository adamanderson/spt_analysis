import os
import numpy as np

logdir = '/scratch/adama/condor_logs/20210407_multiband_limits/test_freq=0.13_amp=0.006_ra0hdec-67.25'
output_dir = '/sptgrid/user/adama/20210407_multiband_limits/test_freq=0.13_amp=0.006_ra0hdec-67.25'
axion_code_dir = '/home/adama/SPT/spt3g_software/scratch/kferguson/axion_oscillation'
script = 'fit_oscillation.py'
angles_dir = '/sptgrid/user/adama/20210407_multiband_limits/angle_sims/'
angles_file = 'sim_angles_20210809_180118_seed=2_freq=0.13_amp=0.006_ra0hdec-67.25.pkl' #'merged_angles.pkl'
files_to_transfer = '{}, {}, {}'.format('limits.sh',
                                            os.path.realpath('fit_oscillation.py'),
                                            os.path.realpath('timefittools.py'))
submit_file = 'limits.submit'
mode = 'simulation'

njobs = 200
n_sims_per_job = 50 #20

fmin = 0.01
fmax = 2.0
n_freqs_total = int(np.floor((fmax - fmin) / 5e-4))
split_n_ways = 1
n_freqs_per_split = int(n_freqs_total / split_n_ways)
freqs = np.linspace(fmin, fmax, n_freqs_total)
freq_edges = [[freqs[n_freqs_per_split*j], freqs[n_freqs_per_split*(j+1)-1]] for j in np.arange(0, split_n_ways)]

signal_amp_freq = [(0, 0.01)] #, (0.2, 0.01), (0.2, 0.5), (1.0, 0.01), (1.0, 0.5)]
job_tag_string = 'sim_limits_freq=0.13_amp=0.006_ra0hdec-67.25'
ul = '' #'--upper-limit-cl 0.95'

for freq_range in freq_edges:
    for signal_amp, signal_freq in signal_amp_freq:
        for jjob in range(njobs):
            jobname = '{}_fitfreq={:.5f}-{:.5f}_amp={:.5f}_freq={:.5f}_{}'\
                        .format(job_tag_string, freq_range[0], freq_range[1],
                                signal_amp, signal_freq, jjob)
            jobid = '{}'.format(jobname)

            print('JOB {} {}'.format(jobname, submit_file))
            print('VARS {} JobID=\"{}\"'.format(jobname, jobid))
            print('VARS {} LogDir=\"{}\"'.format(jobname, logdir))
            print('VARS {} Script=\"{}\"'.format(jobname, script))
            print('VARS {} TransferFiles=\"{}\"'.format(jobname, files_to_transfer))
            if mode == 'simulation':
                print('VARS {} Args=\"{} {} {} {} {} {} {} {} {} --outfile {} {} {}\"'\
                    .format(jobname, mode, angles_file, n_sims_per_job,
                            freq_range[0], freq_range[1], n_freqs_per_split,
                            signal_amp, signal_freq, ul, jobid+'.pkl',
                            os.path.join(angles_dir, angles_file),
                            os.path.join(output_dir, jobid+'.pkl')))
            print('VARS {} NJobs=\"1\"'.format(jobname))
            print('RETRY {} 2'.format(jobname))
