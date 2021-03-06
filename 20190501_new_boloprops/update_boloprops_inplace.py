from spt3g import core, calibration
import copy

bp_filenames = [{'old': '/spt/user/production/calibration/boloproperties/31000000.g3',
                 'new': '31000000_new.g3'},
                {'old': '/spt/user/production/calibration/boloproperties/32512242.g3',
                 'new': '32512242_new.g3'},
                {'old': '/spt/user/production/calibration/boloproperties/34300000.g3',
                 'new': '34300000_new.g3'}]

fname_nominal_boloprops = '/home/adama/SPT/hardware_maps_southpole/2018/hwm_pole_run2_include_darks_v2/nominal_online_cal.g3'
nominal_boloprops = list(core.G3File(fname_nominal_boloprops))[0]

for fnames_dict in bp_filenames:
    fname_old_boloprops = fnames_dict['old']
    old_bp = list(core.G3File(fname_old_boloprops))[0]
    old_boloprops = copy.deepcopy(old_bp)

    newframe = core.G3Frame(core.G3FrameType.Calibration)
    for field in old_boloprops:
        if field != 'BolometerProperties':
            newframe[field] = old_boloprops[field]
    newframe['BolometerProperties'] = calibration.BolometerPropertiesMap()

    print(len(old_boloprops['BolometerProperties']))

    for bolo in old_boloprops['BolometerProperties'].keys():
        if '/' in bolo:
            newbolo = bolo.split('/')[1]
        else:
            newbolo = bolo

        if newbolo in nominal_boloprops['NominalBolometerProperties'].keys():
            newframe['BolometerProperties'][bolo] = old_boloprops['BolometerProperties'][bolo]
            newframe['BolometerProperties'][bolo].pol_angle = nominal_boloprops['NominalBolometerProperties'][newbolo].pol_angle
            #old_boloprops['BolometerProperties'][bolo].pol_angle = nominal_boloprops['NominalBolometerProperties'][newbolo].pol_angle
        else:
            old_boloprops['BolometerProperties'].pop(bolo)
            print('{} not in new nominal bolometer properties!'.format(bolo))
    print(len(old_boloprops['BolometerProperties']))

    writer = core.G3Writer(fnames_dict['new'])
    writer.Process(newframe)
    writer.Process(core.G3Frame(core.G3FrameType.EndProcessing))
