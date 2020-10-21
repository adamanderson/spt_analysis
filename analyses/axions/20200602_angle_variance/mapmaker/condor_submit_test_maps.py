import os
from glob import glob
from spt3g.cluster.condor_tools import condor_submit
from spt3g import core
import os.path
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dirname', action='store',
                    help='Name of directory in which to store job outputs.')
parser.add_argument('configfile', action='store',
                    help='Name of yaml config file.')
parser.add_argument('obslist', action = 'store',
                    help='Text file containing list of source type and '
                    'observation ID, in format:'
                    '[source] [obsID]')
parser.add_argument('--fullrate', action = 'store_true', default = False)
parser.add_argument('--submit', action = 'store_true')
args = parser.parse_args()

# Requirements
job_ram = 8*core.G3Units.GB
job_disk = 20*core.G3Units.GB

# load source and obsid list from text file
df_obslist = pd.read_csv(args.obslist, sep='\t')
obsids = np.array(df_obslist['obsid'], dtype=str)
sources = np.array(df_obslist['source'], dtype=str)

# paths
dir_label = args.dirname
config_file = args.configfile

args_optional = ['-z', '--config-file {}'.format(os.path.basename(config_file))]
aux_files = [config_file]

user_condor_log_dir = '/scratch/adama/condor_logs/'
user_condor_out_dir = '/sptgrid/user/adama/'
script = os.path.join(os.path.dirname(__file__), 'master_field_mapmaker.py')
infiles = [os.path.join(os.path.dirname(__file__), config_file)]
condor_dir = os.path.join(user_condor_log_dir, dir_label)
out_root = os.path.join(user_condor_out_dir, dir_label)

if args.fullrate:
    bolodata_path = '/spt/data/bolodata/fullrate/'
else:
    bolodata_path = '/spt/data/bolodata/downsampled/'

test = True
if args.submit:
    test = False

for source, obsid in zip(sources, obsids):
    job_name = '{}_{}_{}'.format(dir_label, source, obsid)
    outdir = os.path.join(out_root, source, obsid)
    out_fname = job_name+'.g3.gz'
    
    obsdir = os.path.join(bolodata_path, source, str(obsid))
    data_fnames = glob(os.path.join(obsdir, '0*.g3'))
    aux_files.append(os.path.realpath(os.path.join(obsdir, 'offline_calibration.g3')))
    infiles = sorted(data_fnames)

    print(infiles)

    if all(os.path.exists(fn) for fn in infiles):
        args_in = [os.path.basename(dat) for dat in infiles]
        args_script = '{infiles} -o {outfile}' \
            .format(infiles = ' '.join(args_in), 
                    outfile = out_fname,
                    source = source)
        args_script = ' '.join([args_script] + args_optional)

        cluster, f_submit, f_script = condor_submit(script, create_only=test, args = [args_script],
                                                    log_root = condor_dir, 
                                                    output_root = out_root,
                                                    jobname = job_name,
                                                    grid_proxy = '/home/adama/.globus/grid_proxy',
                                                    input_files = infiles,
                                                    output_files = [out_fname],
                                                    aux_input_files = aux_files,
                                                    request_disk = job_disk,
                                                    request_memory = job_ram,
                                                    clustertools_version = 'py3-v3',
                                                    spt3g_env=True)

        print('Creating job for obsid {}'.format(obsid))

    else:
        print('Not processing observation {}:'.format(obsid))
        for fn in infiles:
            if not os.path.exists(fn):
                print('{} does not exist'.format(fn))

        print('')

