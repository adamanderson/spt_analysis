from spt3g import core, dfmux, calibration
import numpy as np
import adama_utils
import os.path

obslist = [68187179, 64599638, 68994321, 70228757]
calpath = '/spt/user/production/calibration/calframe/RCW38-pixelraster/'
fastpointpath = '/spt/user/production/calibration/RCW38-pixelraster/'

rcw38_abs_cal = {
    90.0*core.G3Units.GHz: 4.0549662e-07*core.G3Units.K,
    150.0*core.G3Units.GHz: 2.5601153e-07*core.G3Units.K,
    220.0*core.G3Units.GHz: 2.8025804e-07*core.G3Units.K
}


class ComputeOpteff(object):
    def __init__(self):
        self.CalibratorResponse = []
        self.boloprops = []
        self.FluxCalibration = []
        self.IntegralFlux = []
    
    def __call__(self, frame):
        # If 'CalibratorResponse' is in the frame, then we must be looking at
        # a calframe and not the RCW38-pixelraster output. Note that this is
        # potentially very confusing because the calframe also contains entries
        # keyed with 'RCW38FluxCalibration', but these are *not* filled with the
        # RCW38 output for the observation of interest. If I understand
        # correctly, they contain the RCW38 information used to calibrate the 
        # preceding very fast point.
        if frame.type == core.G3FrameType.Calibration:
            if 'CalibratorResponse' in frame.keys():
                self.CalibratorResponse = frame['CalibratorResponse']
                self.BolometerProperties = frame['BolometerProperties']
            elif 'RCW38FluxCalibration' in frame.keys():
                self.FluxCalibration = frame['RCW38FluxCalibration']
                self.IntegralFlux = frame['RCW38IntegralFlux']

            if len(self.CalibratorResponse) > 0 and \
               len(self.BolometerProperties) > 0 and \
               len(self.FluxCalibration) > 0 and \
               len(self.IntegralFlux) > 0:
                opteff = core.G3MapDouble()
                for bolo in self.CalibratorResponse.keys():
                    if bolo in self.BolometerProperties.keys() and \
                       bolo in self.FluxCalibration.keys():
                        boloprops = self.BolometerProperties[bolo]
                        cal_response = self.CalibratorResponse[bolo]
                        flux_cal = self.FluxCalibration[bolo]
                        int_flux = self.IntegralFlux[bolo]
                        band = boloprops.band

                        opteff[bolo] = -1*cal_response * flux_cal * int_flux / rcw38_abs_cal[band] / adama_utils.wattsPerKcmb(band)
                newframe = core.G3Frame()
                newframe['opteff'] = opteff
                newframe['BolometerProperties'] = self.BolometerProperties
                return newframe
            else:
                return []

# def compute_opteff(fr):
#     if fr.type == core.G3FrameType.Calibration:
#         fr['opteff'] = core.G3MapDouble()
#         for bolo in fr['CalibratorResponse'].keys():
#             if bolo in fr['BolometerProperties'] and \
#                'RCW38FluxCalibration' in fr.keys() and \
#                bolo in fr['RCW38FluxCalibration']:
#                 boloprops = fr['BolometerProperties'][bolo]
#                 cal_response = fr['CalibratorResponse'][bolo]
#                 flux_cal = fr['RCW38FluxCalibration'][bolo]
#                 int_flux = fr['RCW38IntegralFlux'][bolo]
#                 band = boloprops.band

#                 fr['opteff'][bolo] = -1*cal_response * flux_cal * int_flux / rcw38_abs_cal[band] / adama_utils.wattsPerKcmb(band)

for obs in obslist:
    if os.path.exists('{}/{}.g3'.format(calpath, obs)):
        pipe = core.G3Pipeline()
        pipe.Add(core.G3Reader, filename=['{}/{}.g3'.format(calpath, obs),
                                          '{}/{}.g3'.format(fastpointpath, obs)])
        pipe.Add(core.Dump)
        pipe.Add(ComputeOpteff)
        pipe.Add(core.G3Writer, filename='opteff_stdproc_{}.g3'.format(obs))
        pipe.Run()
