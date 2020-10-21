# Simple script for computing the std's of scale pixel values, and performing a
# KS test to check the distribution for gaussianity.

import os.path
from spt3g import maps, core
from glob import glob
import numpy as np
import pickle
import re 
from scipy.stats import kstest
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('group', choices=['2018ee', '2019'])
parser.add_argument('--res', choices=[2, 4], type=int,
                    default=None,
                    help='resolution [arcmin]')
parser.add_argument('--hpf', choices=[100, 200, 300, 400], type=int,
                    default=None,
                    help='high-pass filter cutoff [ell]')
parser.add_argument('--lpf', choices=[3300, 5000, 6600, 8000, 10000], type=int,
                    default=None,
                    help='low-pass filter cutoff [ell]')
parser.add_argument('--commonmode', action='store_true',
                    help='use common-mode filter')
args = parser.parse_args()

if args.group == '2018ee' and \
   (args.res is not None or args.hpf is not None or args.lpf is not None):
    raise argparse.ArgumentError('Requesting 2018 data with mapmaking options. '
                                 'No options available with 2018 data.')
if args.group == '2019' and \
   (args.res is None or args.hpf is None or args.lpf is None):
    raise argparse.ArgumentError('Requesting 2019 data without mapmaking options. '
                                 'Must specify mapmaking options with 2019 data.')

if args.group == '2018ee':
    coadd_dir = '/sptlocal/user/kferguson/full_daniel_maps/'
    coadd_fnames = glob(os.path.join(coadd_dir, '*.g3.gz'))
    out_fname = 'weights_var_factors_{}.pkl'.format(args.group)
elif args.group == '2019':
    if args.commonmode:
        cm_string = '_cm_by_wafer'
    else:
        cm_string = ''

    coadd_dir = '/sptlocal/user/adama/axions/axion_2019_mapmaking_tests_lrcoadds/'
    glob_path = os.path.join(coadd_dir, 'ra0hdec-*_{}_res_{}_hpf_{}_lpf{}_lrcoadd.g3.gz'.format(args.res, args.hpf, args.lpf, cm_string))
    coadd_fnames = glob(glob_path)
    out_fname = 'weights_var_factors_{}_{}_res_{}_hpf_{}_lpf{}_lrcoadd.pkl'.format(args.group, args.res, args.hpf, args.lpf, cm_string)
else:
    exit()

factor = {}
ks_test = {}


for fname in coadd_fnames:
    print('Processing {}'.format(fname))
    if args.group == '2018ee':
        result = re.match('(.*?)_(.*?)GHz_(.*?)_map.g3.gz', os.path.basename(fname))
        field  = result.group(1)
        band   = int(result.group(2))
        obsid  = int(result.group(3))
    if args.group == '2019':
        result = re.match('(.*?)_(.*?)_(.*).g3.gz', os.path.basename(fname))
        field  = result.group(1)
        obsid   = int(result.group(2))
        
    if field not in factor:
        factor[field] = {}
    if field not in ks_test:
        ks_test[field] = {}
    
    if obsid not in factor[field]:
        factor[field][obsid] = {}
    if obsid not in ks_test[field]:
        ks_test[field][obsid] = {}

    if args.group == '2018ee':
        real_data = list(core.G3File(fname))[0]
        maps.RemoveWeights(real_data)

        q_arr = np.array(real_data['Q'])
        q_weight = np.array(real_data['Wpol'].QQ)

        q_arr_finite = q_arr[np.isfinite(q_arr)]
        q_weight_finite = q_weight[np.isfinite(q_arr)]

        factor[field][obsid][band] = np.std(q_arr_finite * np.sqrt(q_weight_finite))

        ks_result = kstest(q_arr_finite * np.sqrt(q_weight_finite) / factor[field][obsid][band],
                           cdf='norm')
        ks_test[field][obsid][band] = ks_result
    elif args.group == '2019':
        datafile = list(core.G3File(fname))
        for jframe, band in zip([5,6,7], [90, 150, 220]):
            real_data = datafile[jframe]
            maps.RemoveWeights(real_data)

            q_arr = np.array(real_data['Q'])
            q_weight = np.array(real_data['Wpol'].QQ)

            q_arr_finite = q_arr[np.isfinite(q_arr)]
            q_weight_finite = q_weight[np.isfinite(q_arr)]

            factor[field][obsid][band] = np.std(q_arr_finite * np.sqrt(q_weight_finite))

            ks_result = kstest(q_arr_finite * np.sqrt(q_weight_finite) / factor[field][obsid][band],
                               cdf='norm')
            ks_test[field][obsid][band] = ks_result

    with open(out_fname, 'wb') as f:
        save_dict = {'std_factor': factor,
                     'ks_result': ks_test}
        pickle.dump(save_dict, f)
