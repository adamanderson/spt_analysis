from spt3g import core, std_processing, calibration, todfilter, mapmaker, dfmux
from spt3g.std_processing.gainmatching import match_gains, difference_pairs, sum_pairs
from spt3g.calibration.template_groups import get_template_groups
from spt3g.todfilter import dftutils
import os.path
import operator
from scipy.optimize import curve_fit
from scipy.signal import welch, periodogram, windows
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('infiles', nargs='+', help='List of input files.')
parser.add_argument('-o', '--output', default='output.g3',
                    help='Name of output file.')
parser.add_argument('--gain-match', action='store_true',
                    help='Calculate gain-matching coefficients.')
parser.add_argument('--per-bolo-asd', action='store_true',
                    help='Calculate and store ASDs for individual bolometers.')
parser.add_argument('--sum-pairs', action='store_true',
                    help='Calculate the pair-summed ASD.')
parser.add_argument('--diff-pairs', action='store_true',
                    help='Calculate the pair-differenced ASD.')
parser.add_argument('--per-pair-asd', action='store_true',
                    help='If pair sum or difference timestreams are calculated, '
                    'then also calculate the ASD.')
parser.add_argument('--average-asd', action='store_true',
                    help='Calculate averaged ASD.')
parser.add_argument('--fit-asd', action='store_true',
                    help='Fit the averaged ASDs.')
parser.add_argument('--units', choices=['temperature', 'current'],
                    default='temperature')
parser.add_argument('--poly-order', default=None, type=int,
                    help='Order of poly filter to apply.')
parser.add_argument('--fit-readout-model', action='store_true',
                    help='Fit ASDs to a noise model that consists of a white '
                    'noise floor with a 1/f component and no photon noise.')
parser.add_argument('--calc-avg-current', action='store_true',
                    help='Calculate the average current of each bolometer '
                    'group and store it in the output frame.')
parser.add_argument('--group-by-wafer', action='store_true',
                    help='When computing PSD averages, group by wafer.')
parser.add_argument('--group-by-band', action='store_true',
                    help='When computing PSD averages, group by band.')
parser.add_argument('--group-by-squid', action='store_true',
                    help='When computing PSD averages, group by squid.')
args = parser.parse_args()

if not args.sum_pairs and not args.diff_pairs and args.per_pair_asd:
    parser.error('--per-pair-asd can only be used if --sum-pairs or '
                 '--diff-pairs is selected.')


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
    return np.sqrt(readout + (A * (x)**(-1*alpha)) + photon / (1 + 2*np.pi*((x*tau)**2)))
def full_readout_model(x, readout, A, alpha):
    return np.sqrt(readout + (A * (x)**(-1*alpha)))

@core.scan_func_cache_data(bolo_props = 'BolometerProperties',
                           wiring_map = 'WiringMap')
def calc_avg_current(frame, ts_key='TimestreamAmps',
                     output_key='AvgCurrent', bolo_props=None, wiring_map=None,
                     average_per_wafer=True, average_per_band=True, average_per_squid=False):
    if frame.type == core.G3FrameType.Scan and \
       ts_key in frame.keys() and bolo_props is not None:
        pixel_tgroups = get_template_groups(bolo_props, wiring_map=wiring_map,
                                            per_band = average_per_band,
                                            per_wafer = average_per_wafer,
                                            per_squid = average_per_squid,
                                            include_keys = True)

        frame[output_key] = core.G3MapDouble()

        n_bolos = {}
        for group_key, bolonames in pixel_tgroups.items():            
            for bolo in bolonames:
                if bolo in frame[ts_key].keys():
                    current = np.median(frame[ts_key][bolo])
                    
                    # cut psds that are not finite or are zero
                    if np.isfinite(current) & (current>0):
                        if group_key not in frame[output_key].keys():
                            n_bolos[group_key] = 0
                            frame[output_key][group_key] = current
                        else:
                            frame[output_key][group_key] += current
                        n_bolos[group_key] += 1

        for group_key in n_bolos.keys():
            frame[output_key][group_key] /= float(n_bolos[group_key])


def fit_asd(frame, asd_key='InputPSD', params_key='PSDFitParams',
            min_freq=0, max_freq=60, params0=(200**2, 10**2, 2, 400**2, 0.01),
            readout_model=False):
    if frame.type == core.G3FrameType.Scan and asd_key in frame.keys():
        for group in frame[asd_key].keys():
            if group != 'frequency':
                if params_key not in frame.keys():
                    frame[params_key] = core.G3MapVectorDouble()

                f = np.array(frame[asd_key]['frequency']) / core.G3Units.Hz
                asd = np.array(frame[asd_key][group])

                try:
                    if readout_model:
                        par, cov = curve_fit(full_readout_model,
                                             f[(f>min_freq) & (f<max_freq)],
                                             asd[(f>min_freq) & (f<max_freq)],
                                             bounds=([0, 0, 0],
                                                     [np.inf, np.inf, np.inf]),
                                             p0=params0)
                    else:
                        par, cov = curve_fit(noise_model,
                                             f[(f>min_freq) & (f<max_freq)],
                                             asd[(f>min_freq) & (f<max_freq)],
                                             bounds=([0, 0, 0, 0, 0],
                                                     [np.inf, np.inf, np.inf, np.inf, 0.1]),
                                             p0=params0)
                except: 
                    par = []
                # except RuntimeError:
                #     par = []

                frame[params_key][group] = par


def calc_asd(frame, ts_key, asd_key='ASD', units='temperature'):
    if frame.type == core.G3FrameType.Scan and \
       ts_key in frame.keys():
        if units == 'temperature':
            # 1/rt(2) for uK rtHz to uK rtsec
            units_factor = (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec) * np.sqrt(2.)) 
        elif units == 'current':
            # rt(2) because `dfmux.ConvertTimestreamUnits` converts to pA_RMS
            # when using current units (see spt3g_software/calibration/python/noise_analysis.py)
            # for more info on this annoying convention.
            units_factor = (core.G3Units.amp*1e-12 / np.sqrt(core.G3Units.Hz)) / np.sqrt(2.)

        asd_integral_key = asd_key + '_integral'
        frame[asd_integral_key] = core.G3MapDouble()

        frame[asd_key] = core.G3MapVectorDouble()
        ts = frame[ts_key]
        psds, freqs = dftutils.get_psd_of_ts_map(ts, pad=False, window_function=windows.boxcar)
        for bolo in psds.keys():
            frame[asd_key][bolo] = np.sqrt(psds[bolo]) / units_factor

        for bolo in psds.keys():
            frame[asd_integral_key][bolo] = np.sum(np.array(psds[bolo])[(freqs>0.01) & (freqs<0.5)])

        frame[asd_key]['frequency'] = freqs


@core.scan_func_cache_data(bolo_props = 'BolometerProperties',
                           wiring_map = 'WiringMap')
def average_asd(frame, ts_key, avg_psd_key='AverageASD', bolo_props=None, wiring_map=None,
                units='temperature',
                average_per_wafer=True, average_per_band=True, average_per_squid=False):
    if frame.type == core.G3FrameType.Scan and \
       ts_key in frame.keys() and bolo_props is not None:
        bolo_tgroups = get_template_groups(bolo_props, wiring_map=wiring_map,
                                            per_band = average_per_band,
                                            per_wafer = average_per_wafer,
                                            per_squid = average_per_squid,
                                            include_keys = True)
        pixel_tgroups = get_template_groups(bolo_props, per_band = True,
                                            per_pixel = True, per_wafer = True,
                                            include_keys=True)

        if units == 'temperature':
            # 1/rt(2) for uK rtHz to uK rtsec
            units_factor = (core.G3Units.microkelvin * np.sqrt(core.G3Units.sec) * np.sqrt(2.)) 
        elif units == 'current':
            # rt(2) because `dfmux.ConvertTimestreamUnits` converts to pA_RMS
            # when using current units (see spt3g_software/calibration/python/noise_analysis.py)
            # for more info on this annoying convention.
            units_factor = (core.G3Units.amp*1e-12 / np.sqrt(core.G3Units.Hz)) / np.sqrt(2.)

        frame[avg_psd_key] = core.G3MapVectorDouble()

        ts = frame[ts_key]
        psds, freqs = dftutils.get_psd_of_ts_map(ts, pad=False, window_function=windows.boxcar)
        n_pairs = {}
        for group_key, bolonames in bolo_tgroups.items():            
            for bolo in bolonames:
                # pair-sum/diff are keyed by pixel name, while timestreams are
                # keyed by bolo name; need to handle both possibilities here.
                pixel_name = [k for k in pixel_tgroups.keys() if bolo in pixel_tgroups[k]][0]
                psd_key = None
                if bolo in frame[ts_key].keys():
                    psd_key = bolo
                elif pixel_name in frame[ts_key].keys():
                    psd_key = pixel_name

                if psd_key is not None:
                    f_pg = freqs
                    psd_pg = psds[psd_key]
                    asd_pg = np.sqrt(psd_pg) / units_factor
                    
                    # cut psds that are not finite or are zero
                    if np.all(np.isfinite(asd_pg) & (asd_pg>0)):
                        if group_key not in frame[avg_psd_key].keys():
                            n_pairs[group_key] = 0
                            frame[avg_psd_key][group_key] = asd_pg
                        else:
                            frame[avg_psd_key][group_key] += asd_pg
                        n_pairs[group_key] += 1
                        frame[avg_psd_key]['frequency'] = f_pg

        for group_key in n_pairs.keys():
            frame[avg_psd_key][group_key] /= float(n_pairs[group_key])



# internal names to handle optional filtering        
if args.poly_order:
    post_poly_ts_key = 'RawTimestreams_I_Filtered'
else:
    post_poly_ts_key = 'RawTimestreams_I'


pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=args.infiles)

if args.calc_avg_current:
    pipe.Add(dfmux.ConvertTimestreamUnits, Input='RawTimestreams_I',
             Output='TimestreamsAmps', Units=core.G3TimestreamUnits.Current)
    pipe.Add(calc_avg_current, ts_key = 'TimestreamsAmps',
             output_key='AvgCurrent',
             average_per_wafer=args.group_by_wafer,
             average_per_band=args.group_by_band,
             average_per_squid=args.group_by_squid)
    pipe.Add(core.Delete, keys=['TimestreamsAmps'])
    pipe.Add(core.Dump)

if args.poly_order:
    pipe.Add(mapmaker.TodFiltering,
             # filtering options
             poly_order = args.poly_order,
             ts_in_key = 'RawTimestreams_I',
             ts_out_key = post_poly_ts_key)

if args.units == 'current':
    pipe.Add(dfmux.ConvertTimestreamUnits, Input=post_poly_ts_key,
             Output='TimestreamsAmps', Units=core.G3TimestreamUnits.Current)
    if args.gain_match:
        pipe.Add(match_gains, ts_key = 'TimestreamsAmps', flag_key=None,
                 gain_match_key = 'GainMatchCoeff', freq_range=[0.01, 0.1])
    ts_data_key = 'TimestreamsAmps'

elif args.units == 'temperature':
    pipe.Add(std_processing.flagsegments.FieldFlaggingPreKcmbConversion,
             flag_key = 'Flags', ts_key = 'RawTimestreams_I')
    pipe.Add(std_processing.CalibrateRawTimestreams,
             i_data_key = post_poly_ts_key,
             output = 'CalTimestreams')
    pipe.Add(std_processing.flagsegments.FieldFlaggingPostKcmbConversion,
             flag_key = 'Flags', ts_key = 'CalTimestreams')
    if args.gain_match:
        pipe.Add(match_gains, ts_key = 'CalTimestreams', flag_key='Flags',
                 gain_match_key = 'GainMatchCoeff', freq_range=[0.01, 0.1])
    else:
        pipe.Add(match_gains, ts_key = 'CalTimestreams', flag_key='Flags',
                 gain_match_key = 'GainMatchCoeff', manual_gain_factor=(1.0, 1.0))
    ts_data_key = 'CalTimestreams'

pipe.Add(core.Dump)

if args.per_bolo_asd:
    pipe.Add(calc_asd, ts_key = ts_data_key, asd_key='ASD', units=args.units)
    if args.fit_asd:
        if args.fit_readout_model:
            pipe.Add(fit_asd, asd_key='ASD', params_key='ASDFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1), readout_model=True)
        else:
            pipe.Add(fit_asd, asd_key='ASD', params_key='ASDFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1, 400**2, 0.01))

if args.sum_pairs:
    pipe.Add(sum_pairs, ts_key = ts_data_key,
             gain_match_key = 'GainMatchCoeff',
             pair_diff_key='PairSumTimestreams')
    if args.average_asd:
        pipe.Add(average_asd, ts_key='PairSumTimestreams', avg_psd_key='AverageASDSum', units=args.units)
        pipe.Add(core.Delete, keys='PairSumTimestreams')
    if args.fit_asd:
        if args.fit_readout_model:
            pipe.Add(fit_asd, asd_key='AverageASDSum', params_key='AverageASDSumFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1), readout_model=True)
        else:
            pipe.Add(fit_asd, asd_key='AverageASDSum', params_key='AverageASDSumFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 2, 400**2, 0.01))
    if args.per_pair_asd:
        pipe.Add(calc_asd, ts_key = ts_data_key, asd_key='ASDSum', units=args.units)

if args.diff_pairs:
    pipe.Add(difference_pairs, ts_key = ts_data_key,
             gain_match_key = 'GainMatchCoeff',
             pair_diff_key='PairDiffTimestreams')
    if args.average_asd:
        pipe.Add(average_asd, ts_key='PairDiffTimestreams', avg_psd_key='AverageASDDiff', units=args.units)
        pipe.Add(core.Delete, keys='PairDiffTimestreams')
    if args.fit_asd:
        if args.fit_readout_model:
            pipe.Add(fit_asd, asd_key='AverageASDDiff', params_key='AverageASDDiffFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1), readout_model=True)
        else:
            pipe.Add(fit_asd, asd_key='AverageASDDiff', params_key='AverageASDDiffFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1, 400**2, 0.01))
    if args.per_pair_asd:
        pipe.Add(calc_asd, ts_key = ts_data_key, asd_key='ASDDiff', units=args.units)

if args.average_asd:
    pipe.Add(average_asd, ts_key=ts_data_key, avg_psd_key='AverageASD', units=args.units,
             average_per_wafer=args.group_by_wafer,
             average_per_band=args.group_by_band,
             average_per_squid=args.group_by_squid)

    if args.fit_asd:
        if args.fit_readout_model:
            pipe.Add(fit_asd, asd_key='AverageASD', params_key='AverageASDFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1), readout_model=True)
        else:
            pipe.Add(fit_asd, asd_key='AverageASD', params_key='AverageASDFitParams',
                     min_freq=0.001, max_freq=60, params0=(200**2, 10**2, 1, 400**2, 0.01))
        

pipe.Add(cleanup, to_save=['GainMatchCoeff',
                           'AverageASDSum', 'AverageASDSumFitParams',
                           'AverageASDDiff', 'AverageASDDiffFitParams',
                           'AverageASD', 'AverageASDFitParams', 'AvgCurrent',
                           'ASD', 'ASDFitParams'])
pipe.Add(core.G3Writer, filename=args.output)
pipe.Run(profile=True)
