from spt3g import core, calibration, dfmux
import pydfmux

old_bp_filenames = ['/mnt/ceph/srm/spt3g/user/production/calibration/boloproperties/40000000.g3',
                    '/mnt/ceph/srm/spt3g/user/production/calibration/boloproperties/60000000.g3']
hwm_filenames = ['/home/adama/SPT/hardware_maps_southpole/2018/hwm_pole_run2_post_event_5/hwm.yaml',
                '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml']
new_bp_filenames = ['40000000.g3',
                    '60000000.g3']

for old_bp_filename, hwm_filename, new_bp_filename in zip(old_bp_filenames, hwm_filenames, new_bp_filenames):
    bp_frame = list(core.G3File(old_bp_filename))[0]
    hwm = pydfmux.load_session(open(hwm_filename))['hardware_map']
    bolos = hwm.query(pydfmux.Bolometer)

    pol_angles = {}
    for b in bolos:
        if b.polarization_angle == -9999:
            pol_angles[b.name] = float('nan')
        else:
            pol_angles[b.name] = b.polarization_angle/core.G3Units.deg
    coupling_dict = {'optical':calibration.BolometerCouplingType.Optical,
                     'dark_xover':calibration.BolometerCouplingType.DarkCrossover,
                     'dark_tres':calibration.BolometerCouplingType.DarkTermination,
                     'resistor':calibration.BolometerCouplingType.Resistor}

    bp_frame.pop('BolometerProperties')
    bp_frame['BolometerProperties'] = calibration.BolometerPropertiesMap()
    for bolo in bolos:
        bp_frame['BolometerProperties'][bolo.name] = calibration.BolometerProperties()
        bp_frame['BolometerProperties'][bolo.name].coupling = coupling_dict[bolo.coupling]
        if coupling_dict[bolo.coupling] == calibration.BolometerCouplingType.Resistor:
            bp_frame['BolometerProperties'][bolo.name].band = float('nan')
            bp_frame['BolometerProperties'][bolo.name].physical_name = bolo.physical_name
            bp_frame['BolometerProperties'][bolo.name].pixel_id = 'N/A'
            bp_frame['BolometerProperties'][bolo.name].pixel_type = 'N/A'
            bp_frame['BolometerProperties'][bolo.name].pol_angle = float('nan')
            bp_frame['BolometerProperties'][bolo.name].pol_efficiency = float('nan')
            bp_frame['BolometerProperties'][bolo.name].wafer_id = bolo.wafer.name
            bp_frame['BolometerProperties'][bolo.name].x_offset = float('nan')
            bp_frame['BolometerProperties'][bolo.name].y_offset = float('nan')
        else:
            bp_frame['BolometerProperties'][bolo.name].band = bolo.observing_band * core.G3Units.GHz
            bp_frame['BolometerProperties'][bolo.name].physical_name = bolo.physical_name
            bp_frame['BolometerProperties'][bolo.name].pixel_id = str(int(bolo.pixel))
            bp_frame['BolometerProperties'][bolo.name].pixel_type = bolo.pixel_type
            bp_frame['BolometerProperties'][bolo.name].pol_angle = bolo.polarization_angle * core.G3Units.deg
            bp_frame['BolometerProperties'][bolo.name].pol_efficiency = 1.0
            bp_frame['BolometerProperties'][bolo.name].wafer_id = bolo.wafer.name
            bp_frame['BolometerProperties'][bolo.name].x_offset = bolo.x_mm
            bp_frame['BolometerProperties'][bolo.name].y_offset = bolo.y_mm
        

    for bolo in bp_frame["BolometerProperties"].keys():
        if bolo in pol_angles.keys():
            bp_frame["BolometerProperties"][bolo].pol_angle = pol_angles[bolo]
        else:
            print('Cannot find bolometer {} in hardware map. Removing'.format(bolo))
            bp_frame["BolometerProperties"].pop(bolo)

    writer = core.G3Writer(new_bp_filename)
    writer.Process(bp_frame)
    writer.Process(core.G3Frame(core.G3FrameType.EndProcessing))
