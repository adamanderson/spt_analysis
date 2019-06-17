from spt3g import core, std_processing, calibration, todfilter, mapmaker, dfmux
from spt3g.std_processing.gainmatching import match_gains, difference_pairs, sum_pairs
from spt3g.calibration.template_groups import get_template_groups
from spt3g.todfilter import dftutils
import os.path
import operator
from scipy.optimize import curve_fit
from scipy.signal import welch, periodogram
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('infiles', nargs='+', help='List of input files.')
parser.add_argument('-o', '--output', default='output.g3',
                    help='Name of output file.')
args = parser.parse_args()


def cleanup(frame, to_save=[]):
    if frame.type == core.G3FrameType.Scan:
        for key in frame:
            if key not in to_save:
                frame.pop(key)
        return frame
    elif frame.type == core.G3FrameType.EndProcessing:
        return frame
    elif frame.type == core.G3FrameType.Calibration:
        return frame
    else:
        return []

def readout_noise(x, readout):
    return readout*np.ones(len(x))
def photon_noise(x, photon, tau):
    return photon / np.sqrt(1 + 2*np.pi*((x*tau)**2))
def atm_noise(x, A, alpha):
    return A * (x)**(-1*alpha)
def noise_model(x, readout, A, alpha, photon, tau):
    return np.sqrt(readout**2 + (A * (x)**(-1*alpha))**2 + photon**2 / (1 + 2*np.pi*((x*tau)**2)))

def fit_asd(frame, asd_key='InputPSD', params_key='PSDFitParams',
            min_freq=0, max_freq=60):
    if frame.type == core.G3FrameType.Scan and asd_key in frame.keys():
        for group in frame[asd_key].keys():
            if group != 'frequency':
                if params_key not in frame.keys():
                    frame[params_key] = core.G3MapVectorDouble()

                f = np.array(frame[asd_key]['frequency']) / core.G3Units.Hz
                asd = np.array(frame[asd_key][group])

                try:
                    par, cov = curve_fit(noise_model,
                                         f[(f>min_freq) & (f<max_freq)],
                                         asd[(f>min_freq) & (f<max_freq)],
                                         bounds=(0, np.inf),
                                         p0=(200, 10, 1, 400, 0.01))
                except RuntimeError:
                    par = []

                frame[params_key][group] = par


@core.scan_func_cache_data(bolo_props = 'BolometerProperties')
def average_asd(frame, ts_key, avg_psd_key='AverageASD', bolo_props=None):
    if frame.type == core.G3FrameType.Scan and \
       ts_key in frame.keys() and bolo_props is not None:
        pixel_tgroups = get_template_groups(bolo_props,
                                            per_band = True, per_pixel = True,
                                            per_wafer = True, include_keys = True)
        wafers = np.unique([bolo_props[bolo].wafer_id for bolo in bolo_props.keys()])
        wafers = wafers[wafers!='']
        bands = np.unique([bolo_props[bolo].band/core.G3Units.GHz for bolo in bolo_props.keys()])
        bands = bands[bands>0]
        group_keys = ['{:.1f}_{}'.format(band, wafer) for band in bands for wafer in wafers]

        frame[avg_psd_key] = core.G3MapVectorDouble()

        ts = frame[ts_key]
        psds, freqs = dftutils.get_psd_of_ts_map(ts, pad=False)

        for wafer in wafers:
            for band in bands:
                group_key = '{:.1f}_{}'.format(band, wafer)
                n_pairs = 0

                for pixel_id in pixel_tgroups:
                    pixel_band = float(pixel_id.split('_')[0])
                    pixel_wafer = pixel_id.split('_')[1]
                    if pixel_band == band and pixel_wafer == wafer and \
                       pixel_id in frame[ts_key].keys():
                        f_pg = freqs
                        psd_pg = psds[pixel_id]
                        asd_pg = np.sqrt(psd_pg) / \
                                 (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec)) / \
                                 np.sqrt(2.) # rt(2) for uK rtHz to uK rtsec
                        if group_key not in frame[avg_psd_key].keys():
                            frame[avg_psd_key][group_key] = core.G3MapVectorDouble()
                            frame[avg_psd_key][group_key] = asd_pg
                        else:
                            frame[avg_psd_key][group_key] += asd_pg
                        n_pairs += 1
                        frame[avg_psd_key]['frequency'] = f_pg

                if n_pairs > 0:
                    frame[avg_psd_key][group_key] /= float(n_pairs)
                    frame[avg_psd_key][group_key] /= np.sqrt(2.) # rt(2) to normalize per-pair noise to per-bolo noise
        

pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=args.infiles)
pipe.Add(std_processing.flagsegments.FieldFlaggingPreKcmbConversion,
         flag_key = 'Flags', ts_key = 'RawTimestreams_I')

pipe.Add(std_processing.CalibrateRawTimestreams,
        output = 'CalTimestreams')

# pipe.Add(todfilter.util.CutTimestreamsWithoutProperties,
#          input='RawTimestreams_I', output='TimestreamsWithProperties')
# pipe.Add(core.Delete, keys=['RawTimestreams_Q'])
# pipe.Add(dfmux.ConvertTimestreamUnits, Input='TimestreamsWithProperties',
#               Output='TimestreamsWatts', Units=core.G3TimestreamUnits.Power)
# pipe.Add(calibration.ApplyTCalibration, Input='TimestreamsWatts',
#              Output='CalTimestreams', Source=['RCW38', 'MAT5A'],
#              OpacityCorrection=False)

pipe.Add(std_processing.flagsegments.FieldFlaggingPostKcmbConversion,
         flag_key = 'Flags', ts_key = 'CalTimestreams')
pipe.Add(match_gains, ts_key = 'CalTimestreams', flag_key='Flags',
         gain_match_key = 'GainMatchCoeff', freq_range=[0.01, 1.0])

pipe.Add(sum_pairs, ts_key = 'CalTimestreams',
         gain_match_key = 'GainMatchCoeff',
         pair_diff_key='PairSumTimestreams')
pipe.Add(average_asd, ts_key='PairSumTimestreams', avg_psd_key='AverageASDSum')
pipe.Add(fit_asd, asd_key='AverageASDSum', params_key='AverageASDSumFitParams',
         min_freq=0.1, max_freq=60)

pipe.Add(difference_pairs, ts_key = 'CalTimestreams',
         gain_match_key = 'GainMatchCoeff',
         pair_diff_key='PairDiffTimestreams')
pipe.Add(average_asd, ts_key='PairDiffTimestreams', avg_psd_key='AverageASDDiff')
pipe.Add(fit_asd, asd_key='AverageASDDiff', params_key='AverageASDDiffFitParams',
         min_freq=0.1, max_freq=60)

pipe.Add(core.Dump)
pipe.Add(cleanup, to_save=['GainMatchCoeff',
                           'PairSumTimestreams', 'SumASDFitParams', 'AverageASDSum', 'AverageASDSumFitParams',
                           'PairDiffTimestreams', 'DiffASDFitParams', 'AverageASDDiff', 'AverageASDDiffFitParams'])
pipe.Add(core.G3Writer, filename=args.output)
pipe.Run()
