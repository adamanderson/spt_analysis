import os
from glob import glob
from spt3g.cluster.condor_tools import condor_submit
from spt3g import core
import os.path
import argparse
import pandas as pd
import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('filelist', action='store',
                    help='Text file containing list map filenames to use.')
parser.add_argument('coaddfile', action='store',
                    help='Coadd filename.')
parser.add_argument('runlabel', action='store', 
                   help='Label to use for this condor run.')
parser.add_argument('--frequency', action = 'store', default='150',
                    help='Observing frequency of coadd.')
parser.add_argument('--submit', action = 'store_true')
args = parser.parse_args()

# Requirements
job_ram = 2*core.G3Units.GB
job_disk = 20*core.G3Units.GB

# load filename and obsidd list from text file
df_filenames = pd.read_csv(args.filelist, delimiter='\t')
ids = np.array(df_filenames['id'], dtype=str)
filenames = np.array(df_filenames['filename'])

condor_dir = '/scratch/adama/condor_logs/{}/'.format(args.runlabel)
out_root = '/spt/user/adama/{}/'.format(args.runlabel)
job_root = 'pol_angle_fit'
script = '/home/adama/SPT/spt_analysis/analyses/axions/20200216_pol_angles/calc_pol_angle.py'


test = True
if args.submit:
    test = False

for idd, obs_fname in zip(ids, filenames):
    job_name = '{}_{}'.format(job_root, idd)
    outdir = os.path.join(out_root, idd)
    out_fname = '{}_results.pkl'.format(idd)
    infiles = [obs_fname, args.coaddfile]

    if all(os.path.exists(fn) for fn in infiles):
        args_script = [os.path.basename(args.coaddfile),
                       os.path.basename(obs_fname),
                       '--frequency', args.frequency,
                       '--results-filename', out_fname]

        cluster, f_submit, f_script = condor_submit(script, create_only=test, args = args_script,
                                                    log_root = condor_dir, 
                                                    output_root = out_root,
                                                    jobname = job_name,
                                                    grid_proxy = '/home/adama/.globus/grid_proxy',
                                                    extra_requirements = '(GLIDEIN_ResourceName =!= "NPX")')
                                                    input_files = infiles,
                                                    output_files = [out_fname],
                                                    aux_input_files = [],
                                                    request_disk = job_disk,
                                                    request_memory = job_ram,
                                                    clustertools_version = 'py3-v3',
                                                    spt3g_env=True)

    else:
        print('Not processing observation {}:'.format(idd))
        for fn in infiles:
            if not os.path.exists(fn):
                print('{} does not exist'.format(fn))

        print('')

