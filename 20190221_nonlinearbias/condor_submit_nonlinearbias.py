import os
from glob import glob
from spt3g.cluster.condor_tools import condor_submit
from spt3g import core
import numpy as np
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('clustername', action = 'store', type=str,
                    help='String to label processing run.')
parser.add_argument('nskies', action = 'store', type=int,
                    help='Number of skies to simulate.')
parser.add_argument('script', action = 'store', type=str,
                    help='Name of python script to run on the grid.')
parser.add_argument('--submit', action = 'store_true',
                    help='Flag that submits jobs to the grid. Default is to '
                    'generate job files locally only.')
pargs = parser.parse_args()


test = True
if pargs.submit:
    test = False

condor_dir = '/scratch/adama/condor_logs/{}/'.format(pargs.clustername)
out_root = '/spt/user/adama/{}/'.format(pargs.clustername)
sim_skies_path = '/spt/user/arahlin/lenspix_maps/*nside8192*.fits'
sim_skies_fnames = np.sort(glob(sim_skies_path))
nskies = np.min([len(sim_skies_fnames), pargs.nskies])


for jsky in range(nskies):
    sim_num = os.path.basename(sim_skies_fnames[jsky]).split('_')[9]
    jobname = '{}_{}'.format(pargs.clustername, sim_num)

    outdir = os.path.join(out_root, jobname)
    cls_outputfile = '{}.pkl'.format(jobname)
    outfiles = [cls_outputfile]
    infiles = [sim_skies_fnames[jsky]]

    log_dir = os.path.join(condor_dir, jobname)

    cluster, f_submit, f_script = condor_submit(pargs.script, create_only=test,
                                                args = [os.path.basename(sim_skies_fnames[jsky]),
                                                        cls_outputfile,
                                                        '--linear-bias-mag 0.0 1.0 2.0 3.0',
                                                        '--ncalstares 4',
                                                        '--norm-to-unbiased',
                                                        '--fit-cosmology'],
                                                log_root = log_dir, 
                                                output_root = outdir,
                                                jobname = jobname,
                                                grid_proxy = '/home/adama/.globus/grid_proxy',
                                                input_files = infiles,
                                                output_files = outfiles,
                                                request_disk = 6*core.G3Units.GB,
                                                request_memory = 6*core.G3Units.GB,
                                                clustertools_version = 'py3-v3',
                                                user_code = '')


    # '--res 1.0',
    # '--ra-pixels 8000',
    # '--dec-pixels 3000'],
