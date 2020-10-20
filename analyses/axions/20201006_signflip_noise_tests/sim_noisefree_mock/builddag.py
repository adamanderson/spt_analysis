from glob import glob
import os
from spt3g import core, std_processing

fitsflag = '-m /sptgrid/user/adama/20201006_signflip_noise_tests/total/total_150ghz_map_3g_0000.fits'
yamlfile = 'axion_noisefree_mock_july2019_params.yaml'
ptsrcfile = '1500d_ptsrc_and_decrement_list.txt'
logdir = '/scratch/adama/condor_logs/20201006_signflip_noise_tests'
mapmaker = 'master_field_mapmaker_signflip.py'
otherfiles = 'axiontod.py'

simstub_fnames = glob('/sptgrid/user/kferguson/axion_perscan_maps_2019/simstub_*_150GHz*g3.gz')
for fname in simstub_fnames:
    jobstr = os.path.basename(fname).lstrip('simstub_').rstrip('.g3.gz')
    print('JOB noisefree-mock-sims-{} makemanymaps.submit'.format(jobstr))
    print('VARS noisefree-mock-sims-{} InputFiles=\"{}\"'.format(jobstr, fname))
    print('VARS noisefree-mock-sims-{} OutputFiles=\"/sptgrid/user/adama/20201006_signflip_noise_tests/noisefree-mock-sims-{}.g3.gz\"'.format(jobstr, jobstr))
    print('VARS noisefree-mock-sims-{} YAMLFile=\"{}\"'.format(jobstr, yamlfile))
    print('VARS noisefree-mock-sims-{} PtSrcFile=\"{}\"'.format(jobstr, ptsrcfile))
    print('VARS noisefree-mock-sims-{} LogDir=\"{}\"'.format(jobstr, logdir))
    print('VARS noisefree-mock-sims-{} Mapmaker=\"{}\"'.format(jobstr, mapmaker))
    print('VARS noisefree-mock-sims-{} OtherFiles=\"{}\"'.format(jobstr, otherfiles))
    print('VARS noisefree-mock-sims-{} ExtraArgs=\"--sim --config-file {} -z {}\"'.format(jobstr, yamlfile, fitsflag))
    print('VARS noisefree-mock-sims-{} JobID=\"noisefree-mock-sims-{}\"'.format(jobstr, jobstr))
