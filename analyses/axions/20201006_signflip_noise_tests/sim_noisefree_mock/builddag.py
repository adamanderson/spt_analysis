from glob import glob
import os
from spt3g import core, std_processing

nskies = 5
fitsflag = '-m /sptgrid/user/adama/20201006_signflip_noise_tests_2/total/total_150ghz_map_3g_0000.fits'
yamlfile = 'axion_noisefree_mock_july2019_params.yaml'
ptsrcfile = '1500d_ptsrc_and_decrement_list.txt'
logdir = '/scratch/adama/condor_logs/20201006_signflip_noise_tests_2'
mapmaker = 'master_field_mapmaker_signflip.py'
otherfiles = 'axiontod.py'
equal_weights_and_flags = True

simstub_fnames = glob('/sptgrid/user/kferguson/axion_perscan_maps_2019/simstub_*_150GHz*g3.gz')
if equal_weights_and_flags:
    simstub_fnames = [fname for fname in simstub_fnames \
                      if 'no_equal_weights_equal_flags' not in fname]
else:
    simstub_fnames = [fname for fname in simstub_fnames \
                      if 'no_equal_weights_equal_flags' in fname]

for fname in simstub_fnames:
    for jsky in range(nskies):
        jobstr = '{}_{:04d}'.format(os.path.basename(fname).lstrip('simstub_')[:-6], jsky)
        fitsflag = '-m /sptgrid/user/adama/20201006_signflip_noise_tests_2/mock_skies/total/total_150ghz_map_3g_{:04d}.fits'.format(jsky)

        print('JOB noisefree-mock-sims-{} makemanymaps.submit'.format(jobstr))
        print('VARS noisefree-mock-sims-{} InputFiles=\"{}\"'.format(jobstr, fname))
        print('VARS noisefree-mock-sims-{} OutputFiles=\"/sptgrid/user/adama/20201006_signflip_noise_tests_2/noisefree-mock-sims-{}.g3.gz\"'.format(jobstr, jobstr))
        print('VARS noisefree-mock-sims-{} YAMLFile=\"{}\"'.format(jobstr, yamlfile))
        print('VARS noisefree-mock-sims-{} PtSrcFile=\"{}\"'.format(jobstr, ptsrcfile))
        print('VARS noisefree-mock-sims-{} LogDir=\"{}\"'.format(jobstr, logdir))
        print('VARS noisefree-mock-sims-{} Mapmaker=\"{}\"'.format(jobstr, mapmaker))
        print('VARS noisefree-mock-sims-{} OtherFiles=\"{}\"'.format(jobstr, otherfiles))
        print('VARS noisefree-mock-sims-{} ExtraArgs=\"--sim --config-file {} -z {}\"'.format(jobstr, yamlfile, fitsflag))
        print('VARS noisefree-mock-sims-{} JobID=\"noisefree-mock-sims-{}\"'.format(jobstr, jobstr))
