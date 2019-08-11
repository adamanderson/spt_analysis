import os
from glob import glob
from spt3g.cluster.condor_tools import condor_submit
from spt3g import core
import os.path
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obslist', action = 'store',
                    help='Text file containing list of source type and '
                    'observation ID, in format:'
                    '[source] [obsID]')
parser.add_argument('--fullrate', action = 'store_true', default = False)
parser.add_argument('--submit', action = 'store_true')
args = parser.parse_args()

# Requirements
job_ram = 5*core.G3Units.GB
job_disk = 20*core.G3Units.GB

# load source and obsid list from text file
df_obslist = pd.read_csv(args.obslist, sep='\t')
obsids = np.array(df_obslist['obsid'], dtype=str)
sources = np.array(df_obslist['source'], dtype=str)

# dir_label = '20190809_noise_gainmatch_cal'
# args_optional = ['--average-asd', '--fit-asd', '--units', 'temperature', '--poly-order 1',
#                  '--diff-pairs', '--sum-pairs', '--group-by-band', '--group-by-wafer',
#                  '--gain-match']
# dir_label = '20190809_noise_gainmatch_cal'
# args_optional = ['--average-asd', '--fit-asd', '--units', 'temperature', '--poly-order 1',
#                  '--diff-pairs', '--sum-pairs', '--group-by-band', '--group-by-wafer',
#                  '--gain-match', '--per-pair-asd']
# dir_label = '20190809_noise_rcw38_cal'
# args_optional = ['--average-asd', '--fit-asd', '--units', 'temperature', '--poly-order 1',
#                  '--diff-pairs', '--sum-pairs', '--group-by-band', '--group-by-wafer']
# dir_label = '20190809_noise_rcw38_cal'
# args_optional = ['--average-asd', '--fit-asd', '--units', 'temperature', '--poly-order 1',
#                  '--diff-pairs', '--sum-pairs', '--group-by-band', '--group-by-wafer', '--per-pair-asd']
dir_label = '20190811_field_gainmatch_test'
args_optional = ['--average-asd', '--fit-asd', '--units', 'temperature', '--poly-order 1',
                 '--diff-pairs', '--sum-pairs', '--group-by-band', '--group-by-wafer',
                 '--gain-match']

condor_dir = '/scratch/adama/condor_logs/{}/'.format(dir_label)
cal_dir = '/spt/user/production/calibration/calframe/'
if args.fullrate:
    out_root = '/spt/user/adama/{}/fullrate/'.format(dir_label)
    bolodata_path = '/spt/data/bolodata/fullrate/'
else:
    out_root = '/spt/user/adama/{}/downsampled/'.format(dir_label)
    bolodata_path = '/spt/data/bolodata/downsampled/'
job_root = 'gainmatching'
script = '/home/adama/SPT/spt_analysis/20190329_gainmatching/test_gain_match_and_fit.py'


test = True
if args.submit:
    test = False

for source, obsid in zip(sources, obsids):
    job_name = '{}_{}_{}'.format(job_root, source, obsid)
    outdir = os.path.join(out_root, source, obsid)
    
    obsdir = os.path.join(bolodata_path, source, str(obsid))
    data_fnames = glob(os.path.join(obsdir, '0*.g3'))
    cal_fname = os.path.join(cal_dir, source, '{}.g3'.format(obsid))
    infiles = [cal_fname] + sorted(data_fnames)

    if all(os.path.exists(fn) for fn in infiles):
        args_in = [os.path.basename(dat) for dat in infiles]
        args_script = '{infiles} -o {outfile}' \
            .format(infiles = ' '.join(args_in), 
                    outfile = job_name+'.g3',
                    source = source)
        args_script = ' '.join([args_script] + args_optional)

        cluster, f_submit, f_script = condor_submit(script, create_only=test, args = [args_script],
                                                    log_root = condor_dir, 
                                                    output_root = out_root,
                                                    jobname = job_name,
                                                    grid_proxy = '/home/adama/.globus/grid_proxy',
                                                    input_files = infiles,
                                                    output_files = [job_name+'.g3'],
                                                    request_disk = job_disk,
                                                    request_memory = job_ram,
                                                    clustertools_version = 'py3-v3',
                                                    spt3g_env=True)

    else:
        print('Not processing observation {}:'.format(obsid))
        for fn in infiles:
            if not os.path.exists(fn):
                print('{} does not exist'.format(fn))

        print('')

