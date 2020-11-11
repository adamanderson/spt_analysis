from glob import glob
import os
from spt3g import core, std_processing

nskies = 5
per_scan_sims = True
pol_angles = True
equal_weights_and_flags = True

if equal_weights_and_flags:
    yaml_file = 'axion_perscan_map_params.yaml'
    simstub_fnames = glob('/sptgrid/user/kferguson/axion_perscan_maps_2019/simstub_ra0*_150GHz*g3.gz')
    jobtag = ''
else:
    yaml_file = 'axion_perscan_map_params_weightsflagsnotequal.yaml'
    simstub_fnames = glob('/sptgrid/user/kferguson/axion_perscan_maps_2019/simstub_no_equal_weights_equal_flags*_150GHz*g3.gz')
    jobtag += 'weightsflagsnotequal_'

for fname in simstub_fnames:
    for jsky in range(nskies):
        jobstr = '{}_{}{:04d}'.format(os.path.basename(fname).lstrip('simstub_')[:-6], jobtag, jsky)
        signflip_outfile = '/sptgrid/user/adama/20201006_signflip_noise_tests_2/noisefree-sims-{}.g3.gz'.format(jobstr)

        fitsflag = '-m /sptgrid/user/adama/20201006_signflip_noise_tests_2/mock_skies/total/total_150ghz_map_3g_{:04d}.fits'.format(jsky)

        if per_scan_sims:
            print('JOB noisefree-sims-{} makemanymaps.submit'.format(jobstr))
            print('VARS noisefree-sims-{} InputFiles=\"{}\"'.format(jobstr, fname))
            print('VARS noisefree-sims-{} OutputFiles=\"{}\"'.format(jobstr, signflip_outfile))
            print('VARS noisefree-sims-{} ExtraArgs=\"--sim --config-file {} -z {}\"'.format(jobstr, yaml_file, fitsflag))
            print('VARS noisefree-sims-{} JobID=\"noisefree-sims-{}\"'.format(jobstr, jobstr))
            print('VARS noisefree-sims-{} YAMLfile=\"{}\"'.format(jobstr, yaml_file))

        coadd_file = '/sptgrid/user/adama/20201006_signflip_noise_tests_2/noisefree-mock-sims_150GHz_coadd_test_{:04d}.g3.gz'.format(jsky)

        if pol_angles:
            print('JOB calc-polangle-{} calcsignflip.submit'.format(jobstr))
            print('VARS calc-polangle-{} InputFiles=\"{} {}\"'.format(jobstr, signflip_outfile, coadd_file))
            print('VARS calc-polangle-{} OutputFiles=\"/sptgrid/user/adama/20201006_signflip_noise_tests_2/calc-polangle-{}.pkl\"'.format(jobstr, jobstr))
            print('VARS calc-polangle-{} ExtraArgs=\"--nmaps 1000 --mapid *150GHz --keep-mock-cmb\"'.format(jobstr))
            print('VARS calc-polangle-{} JobID=\"calc-polangle-{}\"'.format(jobstr, jobstr))

        if per_scan_sims and pol_angles:
            print('PARENT noisefree-sims-{} CHILD calc-polangle-{}'.format(jobstr, jobstr))
