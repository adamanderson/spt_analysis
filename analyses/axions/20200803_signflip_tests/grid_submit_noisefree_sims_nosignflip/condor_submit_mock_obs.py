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
parser.add_argument('stubfilelist', action='store',
                    help='Name of file containing list of simstub files.')
parser.add_argument('simskyfile', action='store',
                    help='Name of fits file that contains simulated cutsky '
                         'map.')
parser.add_argument('yamlfile', action='store',
                    help='Name of YAML file that has the mock-observing '
                         'parameters.')
parser.add_argument('--submit', action='store_true',
                    help='Actually submit the jobs.')
args = parser.parse_args()

# Requirements
job_ram = 4*core.G3Units.GB
job_disk = 4*core.G3Units.GB

# paths
dir_label = args.dirname
user_condor_log_dir = '/scratch/adama/condor_logs/'
user_condor_out_dir = '/sptgrid/user/adama/'
script = os.path.realpath(os.path.join(os.path.dirname(__file__), 'master_field_mapmaker_signflip.py'))
aux_infiles = [args.yamlfile,
               os.path.realpath(os.path.join(os.path.dirname(__file__), 'axiontod.py'))]
condor_dir = os.path.join(user_condor_log_dir, dir_label)
out_root = os.path.join(user_condor_out_dir, dir_label)

test = True
if args.submit:
    test = False

# parse simstub names
simstub_fnames = np.loadtxt(args.stubfilelist, dtype=str)

for simstub_fname in simstub_fnames:
    job_name = os.path.splitext(os.path.splitext(os.path.basename(simstub_fname))[0])[0]
    job_name = job_name.lstrip('simstub_')

    outfile_name = 'signflip_map_{}.g3.gz'.format(job_name)
    args_script = [os.path.basename(simstub_fname),
                   '--sim',
                   '-z',
                   '--sim-map', os.path.basename(args.simskyfile),
                   '--config-file', os.path.basename(args.yamlfile),
                   '-o', outfile_name]
                   #'--sign-flip-noise']
    #args_script = ' '.join(args_script)
    infiles = [args.simskyfile, simstub_fname]

    cluster, f_submit, f_script = condor_submit(script, create_only=test, args=args_script,
                                                log_root = condor_dir, 
                                                output_root = out_root,
                                                jobname = job_name,
                                                input_files = infiles,
                                                aux_input_files = aux_infiles,
                                                grid_proxy = '/home/adama/.globus/grid_proxy',
                                                output_files = [outfile_name],
                                                request_disk = job_disk,
                                                request_memory = job_ram,
                                                clustertools_version = 'py3-v3',
                                                spt3g_env=True)
    
    print('Creating job for {}'.format(simstub_fname))
