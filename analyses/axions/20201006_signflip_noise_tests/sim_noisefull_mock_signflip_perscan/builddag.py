from glob import glob
import os
from spt3g import core, std_processing

nskies = 5
per_scan_sims = True
pol_angles = True
equal_weights_and_flags = True
logdir = '/scratch/adama/condor_logs/20201006_signflip_noise_tests_2'

if equal_weights_and_flags:
    yaml_file = 'axion_perscan_map_params.yaml'
    data_fnames = glob('/sptgrid/user/kferguson/axion_perscan_maps_2019/ra0hdec-*_150GHz*g3.gz')
    jobtag = ''
else:
    yaml_file = 'axion_perscan_map_params_weightsflagsnotequal.yaml'
    data_fnames = glob('/sptgrid/user/kferguson/axion_perscan_maps_2019/no_equal_weights_equal_flags*_150GHz*g3.gz')
    jobtag = 'weightsflagsnotequal_'

for fname in data_fnames:
    for jsky in range(nskies):
        jobstr = 'noisefull_{}_{}{:04d}'.format(os.path.basename(fname)[:-6], jobtag, jsky)

        signflip_outfile = '/sptgrid/user/adama/20201006_signflip_noise_tests_2/noisefree_perscan_maps_{}.g3.gz'.format(jobstr)

        fitsflag = '-m /sptgrid/user/adama/20201006_signflip_noise_tests_2/mock_skies/total/total_150ghz_map_3g_{:04d}.fits'.format(jsky)

        coadd_file = '/sptgrid/user/adama/20201006_signflip_noise_tests/noisefree-mock-sims-150GHz-coadd.g3.gz'
        obsid = fname[-14:-6]

        mockobs_matches = glob('/sptgrid/user/adama/20201006_signflip_noise_tests/noisefree-mock-sims-ra*{}.g3.gz'.format(obsid))
        if len(mockobs_matches) == 1:
            mockobs_fname = mockobs_matches[0]

            print('JOB calc-polangle-{} calcsignflip.submit'.format(jobstr))
            print('VARS calc-polangle-{} InputFiles=\"{} {} {}\"'.format(jobstr, fname, coadd_file, mockobs_fname))
            print('VARS calc-polangle-{} OutputFiles=\"/sptgrid/user/adama/20201006_signflip_noise_tests/calc-polangle-{}.pkl\"'.format(jobstr, jobstr))
            print('VARS calc-polangle-{} ExtraArgs=\"--nmaps 1000 --mapid *150GHz\"'.format(jobstr))
            print('VARS calc-polangle-{} JobID=\"calc-polangle-{}\"'.format(jobstr, jobstr))
