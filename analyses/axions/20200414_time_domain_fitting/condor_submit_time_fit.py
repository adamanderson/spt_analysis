import os
from glob import glob
from spt3g.cluster.condor_tools import condor_submit
from spt3g import core
import os.path
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('njobs', action='store', type=int,
                    help='Number of jobs to submit')
parser.add_argument('nsimsperjob', action='store', type=int,
                    help='Number of realizations of fake data to simulate per job.')
parser.add_argument('fmin', action='store', type=float,
                    help='Minimum frequency to simulate.')
parser.add_argument('fmax', action='store', type=float,
                    help='Maximum frequency to simulate.')
parser.add_argument('nfreqs', action='store', type=int,
                    help='Number of frequencies to simulate between fmin and '
                    'fmax.')
parser.add_argument('dirname', action='store',
                    help='Name of directory in which to store job outputs.')
parser.add_argument('--uniform-data', action='store_true',
                    help='Simulate data as uniform in time. Duration and number '
                    'of observations are hardcoded.')
parser.add_argument('--nseasons', action='store', type=int, default=1,
                    help='Number of seasons to simulate. Takes obsids in '
                    'pickle file and concatenates them, offset by 1 year.')
parser.add_argument('--submit', action='store_true',
                    help='Actually submit the jobs.')
args = parser.parse_args()

# Requirements
job_ram = 2*core.G3Units.GB
job_disk = 2*core.G3Units.GB

# paths
dir_label = args.dirname
user_condor_log_dir = '/scratch/adama/condor_logs/'
user_condor_out_dir = '/sptgrid/user/adama/'
script = os.path.join(os.path.dirname(__file__), 'simulate_time_domain.py')
infiles = [os.path.join(os.path.dirname(__file__), 'timefittools.py'),
           os.path.join(os.path.dirname(__file__), 'obsids_1500d_2019.pkl')]
condor_dir = os.path.join(user_condor_log_dir, dir_label)
out_root = os.path.join(user_condor_out_dir, dir_label)


test = True
if args.submit:
    test = False

for jjob in np.arange(args.njobs):
    job_name = 'timefit_job{}'.format(jjob)
    outfile_name = 'sim_output_job{}.pkl'.format(jjob)
    args_optional = ['{}'.format(args.nsimsperjob),
                     '{}'.format(args.fmin),
                     '{}'.format(args.fmax),
                     '{}'.format(args.nfreqs),
                     '--pol-error 1.8',
                     '--nseasons {}'.format(args.nseasons),
                     '--outfile {}'.format(outfile_name)]
    if args.uniform_data:
        args_optional.append('--duration 270')
        args_optional.append('--obs 1729')
    else:
        args_optional.append('--obs {}'.format('obsids_1500d_2019.pkl'))

    args_script = ' '.join(args_optional)

    cluster, f_submit, f_script = condor_submit(script, create_only=test, args = [args_script],
                                                log_root = condor_dir, 
                                                output_root = out_root,
                                                jobname = job_name,
                                                aux_input_files = infiles,
                                                grid_proxy = '/home/adama/.globus/grid_proxy',
                                                output_files = [outfile_name],
                                                request_disk = job_disk,
                                                request_memory = job_ram,
                                                clustertools_version = 'py3-v3',
                                                spt3g_env=True,
                                                new_storage=True)
    
    print('Creating job #{}'.format(jjob))
