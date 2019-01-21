# One of the problems with the noise processing is that, for various technical
# reasons concerning the definition of a `source` in a `schedule`, the 
# autoprocessing script does not choose the correct RCW38 and MAT5A observation
# when calculating the temperature calibration. As an workaround, this script
# builds calframes with the correct calibration.

import os.path
from spt3g import core, dfmux, calibration

noise_calframe_dir = '/spt/user/production/calibration/calframe/noise/'
source_autoproc_dir = '/spt/user/production/calibration/'

cal_obsids = {64576620: {'fast point': 64569859,
                         'very fast point': 64576080},
              64591411: {'fast point': 64584651,
                         'very fast point': 64590871},
              64606397: {'fast point': 64599638,
                         'very fast point': 64605858},
              64685912: {'fast point': 64678194,
                         'very fast point': 64685310},
              64685310: {'fast point': 64693360,
                         'very fast point': 64700476},}


class CalFixer(object):
    def __init__(self):
        self.save_frame = core.G3Frame(core.G3FrameType.Calibration)
        
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Calibration:
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
    rcw38_veryfastpoint_path = os.path.join(source_autoproc_dir,
                                            'RCW38',
                                            '{}.g3'.format(cal_obsids[noise_obsid]['very fast point']))
    mat5a_veryfastpoint_path = os.path.join(source_autoproc_dir,
                                            'MAT5A',
                                            '{}.g3'.format(cal_obsids[noise_obsid]['very fast point']))
    if os.path.exists(rcw38_fastpoint_path):
        fastpoint_path = rcw38_fastpoint_path
    else:
        fastpoint_path = mat5a_fastpoint_path
    if os.path.exists(rcw38_veryfastpoint_path):
        veryfastpoint_path = rcw38_veryfastpoint_path
    else:
        veryfastpoint_path = mat5a_veryfastpoint_path


    if os.path.exists(noise_calframe_path) and \
       os.path.exists(fastpoint_path) and \
       os.path.exists(veryfastpoint_path):
        pipe = core.G3Pipeline()
        pipe.Add(core.G3Reader, filename=[noise_calframe_path,
                                          fastpoint_path,
                                          veryfastpoint_path])
        pipe.Add(CalFixer)
        pipe.Add(core.G3Writer, filename='calframe_corrected/{}_calframe_corrected.g3'.format(noise_obsid))
        pipe.Run()
    else:
        print('Missing input files. Skipping.')
