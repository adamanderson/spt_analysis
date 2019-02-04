# One of the problems with the noise processing is that, for various technical
# reasons concerning the definition of a `source` in a `schedule`, the 
# autoprocessing script does not choose the correct RCW38 and MAT5A observation
# when calculating the temperature calibration. As an workaround, this script
# builds calframes with the correct calibration.

import os.path
from spt3g import core, dfmux, calibration

noise_calframe_dir = '/spt/user/production/calibration/calframe/noise/'
source_autoproc_dir = '/spt/user/production/calibration/'

cal_obsids = {63084862: {'fast point': 63077834},
              63098693: {'fast point': 63091665},
              63218920: {'fast point': 63211891},
              63227942: {'fast point': 63211891},
              63305224: {'fast point': 63299089},
              63380372: {'fast point': 63386507},
              63640406: {'fast point': 63632925},
              63650590: {'fast point': 63643116},
              63661173: {'fast point': 63653698},
              63689042: {'fast point': 63682012},
              63728180: {'fast point': 63721146},
              64576620: {'fast point': 64569859,
                         'very fast point': 64576080},
              64591411: {'fast point': 64584651,
                         'very fast point': 64590871},
              64606397: {'fast point': 64599638,
                         'very fast point': 64605858},
              64685912: {'fast point': 64678194,
                         'very fast point': 64685310},
              64701072: {'fast point': 64693360,
                         'very fast point': 64700476},
              64716070: {'fast point': 64708356,
                         'very fast point': 64715473},
              65041359: {'fast point': 65033646,
                         'very fast point': 65040762},
              65106264: {'fast point': 65098550,
                         'very fast point': 65105667},
              65118448: {'fast point': 65110734,
                         'very fast point': 65117851},
              65134617: {'fast point': 65126904,
                         'very fast point': 65134020},
              65146903: {'fast point': 65139189,
                         'very fast point': 65146306},}


class CalFixer(object):
    def __init__(self):
        self.save_frame = core.G3Frame(core.G3FrameType.Calibration)
        
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Calibration:
            # if this is an existing calframe, then remove all the keys
            # that we eventually want to replace
            if 'CalibratorResponse' in frame.keys():
                frame.pop('RCW38FluxCalibration')
                frame.pop('MAT5AFluxCalibration')
                frame.pop('RCW38IntegralFlux')
                frame.pop('MAT5AIntegralFlux')
                frame.pop('RCW38SkyTransmission')
                frame.pop('MAT5ASkyTransmission')

            # ruthlessly overwrite all entries in calibration frames
            for key in frame.keys():                    
                if key in self.save_frame.keys():
                    self.save_frame.pop(key)
                self.save_frame[key] = frame[key]

        if frame.type == core.G3FrameType.EndProcessing:
            return [self.save_frame, frame]
        else:
            return []


for noise_obsid in cal_obsids:
    print('Processing calframe for {}'.format(noise_obsid))

    # load the existing calibration info for the noise stare
    noise_calframe_path = os.path.join(noise_calframe_dir,
                                       '{}.g3'.format(noise_obsid))
    rcw38_fastpoint_path = os.path.join(source_autoproc_dir,
                                        'RCW38-pixelraster',
                                        '{}.g3'.format(cal_obsids[noise_obsid]['fast point']))
    mat5a_fastpoint_path = os.path.join(source_autoproc_dir,
                                        'MAT5A-pixelraster',
                                        '{}.g3'.format(cal_obsids[noise_obsid]['fast point']))
    if os.path.exists(rcw38_fastpoint_path):
        fastpoint_path = rcw38_fastpoint_path
    else:
        fastpoint_path = mat5a_fastpoint_path


    if 'very fast point' in cal_obsids[noise_obsid].keys():
        rcw38_veryfastpoint_path = os.path.join(source_autoproc_dir,
                                                'RCW38',
                                                '{}.g3'.format(cal_obsids[noise_obsid]['very fast point']))
        mat5a_veryfastpoint_path = os.path.join(source_autoproc_dir,
                                                'MAT5A',
                                                '{}.g3'.format(cal_obsids[noise_obsid]['very fast point']))
        if os.path.exists(rcw38_veryfastpoint_path):
            veryfastpoint_path = rcw38_veryfastpoint_path
        else:
            veryfastpoint_path = mat5a_veryfastpoint_path
    else:
        veryfastpoint_path = None

    print(noise_calframe_path)
    print(fastpoint_path)
    if os.path.exists(noise_calframe_path) and \
       os.path.exists(fastpoint_path):
        pipe = core.G3Pipeline()
        if veryfastpoint_path == None:
            pipe.Add(core.G3Reader, filename=[noise_calframe_path,
                                              fastpoint_path])
        else:
            pipe.Add(core.G3Reader, filename=[noise_calframe_path,
                                              fastpoint_path,
                                              veryfastpoint_path])
        pipe.Add(CalFixer)
        pipe.Add(core.G3Writer, filename='calframe_corrected/{}_calframe_corrected.g3'.format(noise_obsid))
        pipe.Run()
    else:
        print('Missing input files. Skipping.')
