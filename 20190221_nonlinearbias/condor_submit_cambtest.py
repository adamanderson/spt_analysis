import os
from glob import glob
from spt3g.cluster.condor_tools import condor_submit
from spt3g import core
import numpy as np
import os.path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('jobname', action = 'store', type=str,
                    help='String to label processing run.')
parser.add_argument('script', action = 'store', type=str,
                    help='Name of python script to run on the grid.')
parser.add_argument('--submit', action = 'store_true',
                    help='Flag that submits jobs to the grid. Default is to '
                    'generate job files locally only.')
pargs = parser.parse_args()


test = True
if pargs.submit:
    test = False

condor_dir = '/scratch/adama/condor_logs/{}/'.format(pargs.jobname)
out_root = '/spt/user/adama/{}/'.format(pargs.jobname)

cluster, f_submit, f_script = condor_submit(pargs.script, create_only=test, args = [],
                                            log_root = condor_dir, 
                                            output_root = out_root,
                                            jobname = pargs.jobname,
                                            grid_proxy = '/home/adama/.globus/grid_proxy',
                                            input_files = [], #infiles,
                                            output_files = [], #outfiles,
                                            request_disk = 2*core.G3Units.GB,
                                            request_memory = 2*core.G3Units.GB,
                                            clustertools_version = 'py3-v3',
                                            user_code = '')


