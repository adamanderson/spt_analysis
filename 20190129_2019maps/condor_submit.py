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
parser.add_argument('source', action = 'store', type=str,
                    help='Name of source to process.')
parser.add_argument('obsidlist', action = 'store', type=str,
                    help='File containing list of observation IDs to process.')
parser.add_argument('script', action = 'store', type=str,
                    help='Name of python script to run on the grid.')
parser.add_argument('--fullrate', action = 'store_true',
                    help='Flag that selects to use fullrate data instead of '
                    'downsampled (default).')
parser.add_argument('--submit', action = 'store_true',
                    help='Flag that submits jobs to the grid. Default is to '
                    'generate job files locally only.')
pargs = parser.parse_args()


if pargs.fullrate == True:
    ratestr = 'fullrate'
else:
    ratestr = 'downsampled'

test = True
if pargs.submit:
    test = False

condor_dir = '/scratch/adama/condor_logs/{}/'.format(pargs.jobname)
out_root = '/spt/user/adama/{}/{}/'.format(pargs.jobname, ratestr)

bolodata_path = '/spt/data/bolodata/{}/{}/'.format(ratestr, pargs.source)
caldata_path = '/spt/user/production/calibration/calframe/'

obsids = np.loadtxt(pargs.obsidlist, dtype=int)


for jobs, obs in enumerate(obsids):
    outdir = os.path.join(out_root,str(obs))
    rawdatadir = os.path.join(bolodata_path, str(obs))
    caldatafiles = [os.path.join(caldata_path, pargs.source, '{}.g3'.format(obs))]
    rawdatafiles = glob(os.path.join(rawdatadir, '0*.g3'))
    infiles = caldatafiles + sorted(rawdatafiles)
    outputfile = '{}_output.g3'.format(obs)

    if all(os.path.exists(fn) for fn in infiles):
        args_in = [os.path.basename(dat) for dat in infiles]
        optional_args = '--output {} '.format(outputfile) + \
                        '--source {} '.format(pargs.source) + \
                        '--res 2.0 ' + \
                        '--xlen 100 --ylen 60 --lr'
        args = args_in + [optional_args]
        print(args)
        cluster, f_submit, f_script = condor_submit(pargs.script, create_only=test, args = args,
                                                    log_root = condor_dir, 
                                                    output_root = out_root,
                                                    jobname = str(obs),
                                                    grid_proxy = '/home/adama/.globus/grid_proxy',
                                                    input_files= infiles,
                                                    output_files=[outputfile],
                                                    request_disk=8*core.G3Units.GB,
                                                    request_memory=6*core.G3Units.GB,
                                                    clustertools_version='py3-v3')

    else:
        print('Not processing observation {}:'.format(obs))
        for fn in infiles:
            if not os.path.exists(fn):
                print('{} does not exist'.format(fn))

        print('')

