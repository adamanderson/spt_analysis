'''
Script to make CMB field maps.
'''
import argparse
import os
import yaml
import numpy as np
from spt3g import core, std_processing, mapmaker, calibration, sources
from spt3g import timestreamflagging, todfilter, maps, pointing
import axiontod
import sys

# Usage: master_field_mapmaker.py <input files.g3> -o outputmaps.g3
#            --config-file <config.yaml>

# =============================================================================
# Load in all settings for the map-making pipeline,
# either from command line or configuration yaml if one is given.
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Maps for a CMB field')
cli = parser.add_argument_group('Command Line Inputs', 'These settings are '
                                'specified via the command line only:')
cli.add_argument('input_files', nargs = '+',
                    help = ('The .g3 files containing observation data. The '
                            'offline-calibration.g3 file must be first.'))
cli.add_argument('-o', '--output', default = 'output.g3',
                    help = 'Output filename')
cli.add_argument('-z', '--compress', default=False, action='store_true',
                    help='Store output maps in compressed .g3.gz file')
cli.add_argument('--produce-simstub',default = False, action = 'store_true',
                    help = ('Include this flag to produce a simstub. These '
                            'are used later for mock observations, and include'
                            ' information on flags and pointing. Same location'
                            ' as output, with "simstub-" prepended to file.'))
cli.add_argument('--sim', default = False, action = 'store_true',
                    help = ('Include this flag when mock-observing a '
                            'simulated map using simstubs as input files.'))
cli.add_argument('-m', '--sim-map',
                    help = 'Path to simulated .fits map to mock-observe.')
cli.add_argument('--interp', default = False, action = 'store_true',
                    help = 'Interpolate the input sim map when mock-observing.')
cli.add_argument('--add-ts-noise', default = False, action = 'store_true',
                    help = ('Add noise at the timestream level when '
                            'mock-observing.'))
cli.add_argument('--ts-noise-seed', default = -1, type=int,
                    help = ('Seed for random number generator when adding noise '
                            'at the timestream level in mock observations.  By '
                            'default, will seed based on current time.'))
cli.add_argument('-v', '--verbose', default=False, action = 'store_true',
                    help = 'Print frames to screen.')
cli.add_argument('--config-file',
                    help = ('.yaml file containing map-making parameters. '
                            'These will override command-line settings '
                            'for the options below.'
                            'See default_config.yaml for an example.'))
config = parser.add_argument_group('Config File Inputs', 'These settings '
                                   'are specified in the config file:')
config.add_argument('-s', '--map-source',
                    help = ('Name of source, used to set map center. '
                            'Overrides center ra and dec.'))
config.add_argument('--map-center-ra', type=float, default = 0,
                    help = ('RA center of output-map in degrees. '
                            'Only used if source not specified.'))
config.add_argument('--map-center-dec', type=float, default = -57.5,
                    help = ('Dec center of output-map in degrees. '
                            'Only used if source not specified.'))
config.add_argument('-x', '--map-width', default = 75, type=float,
                    help = 'Output-map width, in degrees.')
config.add_argument('-y', '--map-height', default = 50, type=float,
                    help = 'Output-map height, in degrees.')
config.add_argument('-r', '--map-resolution', default = 2.0, type=float,
                    help = 'Resolution of the output-map, in arcmin.')
config.add_argument('--map-projection', default = 5, type=int,
                    help = ('Projection of the output map. '
                            '0 = SansonFlamsteed (sinusoidal), '
                            '1 = Cartesian, '
                            '2 = Orthographic, '
                            '4 = Stereographic, '
                            '5 = LambertAzimuthalEqualArea, '
                            '9 = BICEP'))
config.add_argument('--temperature-only', default = False,
                    action = 'store_true',
                    help = 'Include this flag to make T-only maps.')
config.add_argument('--add-temperature-only', default = False, action = 'store_true',
                    help = 'Create a T-only map in addition to polarized')
config.add_argument('--wafers-to-exclude', nargs = '+', default = [],
                    help = 'Wafers to exclude from mapmaking.')
config.add_argument('--wafers-to-include', nargs = '+', default=[],
                    help = ('Wafers to include in mapmaking. By default all '
                            'wafers present in BolometerProperties will be used.'))
config.add_argument('--bands-to-use', nargs = '+', default=['90','150','220'],
                    choices=['90GHz', '150GHz', '220GHz', '90', '150', '220'],
                    help = ("Only use detectors in these observing bands. "
                            "Options are '90GHz', '150GHz', and/or '220GHz'"))
config.add_argument('--mask-point-sources', default = False,
                    action = 'store_true')
config.add_argument('--point-source-file', default = None,
                    help = ('Point source file for making a point source mask.'
                            ' File should include INDEX, RA, DEC, RADIUS '
                            'columns. Note: this needs to be the location of '
                            'the file on the machine that runs this script.'))
config.add_argument('--cut-az-unwraps', default = False, action = 'store_true')
config.add_argument('--cut-az-glitches', default = False, action= 'store_true')
config.add_argument('--minnum-bolos-per-scan', type=int, default = 1)
config.add_argument('--apply-common-mode-filter', default = False,
                    action = 'store_true',
                    help = ('Subtract the average signal over a group of '
                            'detectors from each detector in that group. '
                            'If no further options are set, will use average '
                            'signal across all detectors. Options stack: e.g. '
                            '--cm-by-band used with --cm-by-squid will '
                            'construct groups of detectors that have the same '
                            'band and are on the same squid.'))
config.add_argument('--cm-by-wafer', default = False, action = 'store_true')
config.add_argument('--cm-by-band', default = False, action = 'store_true')
config.add_argument('--cm-by-squid', default = False, action = 'store_true')
config.add_argument('--mask-common-mode', default = False, action = 'store_true')
config.add_argument('--poly-order', default = 19, type = int,
                    help = ('Order of the polynomial filter applied to the '
                            'timestreams.'))
config.add_argument('--filters-are-ell-based', default = False,
                    action = 'store_true',
                    help = ('If True, high-pass-cutoff and low-pass-cutoff '
                            'will be specified in ell-space. Otherwise Hz.'))
config.add_argument('--high-pass-cutoff', type = float, default = 300,
                    help = 'High-pass freq of filter applied to timestreams.')
config.add_argument('--low-pass-cutoff', type = float, default = 6600,
                    help = 'Low-pass freq of filter applied to timestreams.')
config.add_argument('--apply-notch-filter',
                    default = False, action = 'store_true',
                    help = ('Subtract best-fit (linear least squares based) '
                            'sine waves at those sample frequency values '
                            'computed by fftfreqs that fall within '
                            'user-specified frequency ranges. These ranges '
                            'should be recorded in a calibration frame '
                            'that is included in an input file.'))
config.add_argument('--weight-low-freq', type = float, default = 1.0,
                    help=("The lower limit of the frequency range, in Hz, used"
                          " to compute the weight of each detector's data."))
config.add_argument('--weight-high-freq', type = float, default = 4.0,
                    help=("The upper limit of the frequency range, in Hz, used"
                          " to compute the weight of each detector's data."))
config.add_argument('--pointing-model', default = 'online',
                    choices = ['online', 'offline','Online','Offline'])
config.add_argument('--split-left-right', default = False,
                    help = ('Split left-going and right-going scans into '
                            'different maps. Can keep only left-going maps or '
                            'only right-going maps by passing "left" or '
                            '"right", respectively.'))
config.add_argument('--split-by-band', default = False, action = 'store_true',
                    help = ('Split data from different observing bands into '
                            'different maps.'))
config.add_argument('--split-by-wafer', default = False, action = 'store_true',
                    help = ('Split data from different wafers into '
                            'different maps.'))
config.add_argument('--sign-flip-noise', action = 'store_true',
                    help = ('Flip signs of scans randomly, enforcing L/R to '
                            'have opposite signs in order to cancel sky '
                            'signals.'))
config.add_argument('--sign-flip-seed', type = int, default = None,
                    help = 'Random seed for sign flips to use in per-scan '
                    'sign-flip noise generation. Passed to '
                    'numpy.random.RandomState. `None` (default) draws seed '
                    'from /dev/urandom.')
config.add_argument('--equal-flags-in-scan-pairs', action='store_true',
                    default=False,
                    help = ('Enforce that the same set of bolometer flags '
                    'are applied for scans in L/R pairs.'))
config.add_argument('--avg-lr-weights', action = 'store_true',
                    help = ('Use the average weights for L-going and R-going '
                    'scans if make per-scan maps.'))

args = parser.parse_args()


# If configuration yaml is specified, load it and pull parameters from there.
if args.config_file is not None:
    settings = yaml.safe_load(open(args.config_file, 'r'))
    for k, v in settings.items():
        setattr(args, k, v)
        
# -----------------------------------------------------------------------------
# Reconfiguring some of the inputs:
# -----------------------------------------------------------------------------
if args.compress and not args.output.endswith('.gz'):
    args.output += '.gz'

if args.map_source not in ['', None]:
    ra, dec = std_processing.sourcemaps.get_source_ra_dec(args.map_source)
else:
    ra = args.map_center_ra * core.G3Units.deg
    dec = args.map_center_dec * core.G3Units.deg

res = args.map_resolution * core.G3Units.arcmin
x_len = int(args.map_width * core.G3Units.deg / res)
y_len = int(args.map_height * core.G3Units.deg / res)

if args.map_projection in maps.MapProjection.names:
    proj = maps.MapProjection.names[args.map_projection]
elif args.map_projection in maps.MapProjection.values:
    proj = maps.MapProjection.values[args.map_projection]
else:
    raise TypeError("%s is not a valid map projection"%args.map_projection)

if not args.filters_are_ell_based:
    hpf = core.G3Units.Hz * args.high_pass_cutoff
    lpf = core.G3Units.Hz * args.low_pass_cutoff
else:
    hpf = args.high_pass_cutoff
    lpf = args.low_pass_cutoff
    
bands = []
for band in args.bands_to_use:
    if 'ghz' not in str(band).lower():
        band=str(int(band))+'GHz'
    bands.append(band)

bad_wafers = []
for wafer in args.wafers_to_exclude:
    if wafer not in [None, '']:
        bad_wafers.append(wafer.capitalize())

good_wafers = []
for wafer in args.wafers_to_include:
    if wafer not in [None, '']:
        good_wafers.append(wafer.capitalize())

# Suppress warnings about timestream missing from scan frame
if args.split_left_right:
    core.set_log_level(core.G3LogLevel.LOG_ERROR, 'MapBinner')

# -----------------------------------------------------------------------------
# Generate map stubs for use later in pipeline.
# -----------------------------------------------------------------------------
map_params = maps.FlatSkyMap(
    x_len = x_len, y_len = y_len, res = res,
    proj = proj, alpha_center = ra, delta_center = dec,
    pol_type = maps.MapPolType.T,
    coord_ref = maps.MapCoordReference.Equatorial)

# Set up an empty map for point source filtering
if args.mask_point_sources and args.point_source_file not in [None, '']:
    ps_params = maps.FlatSkyMap(
        x_len = x_len, y_len = y_len, res = res,
        proj = proj, alpha_center = ra, delta_center = dec,
        pol_type = maps.MapPolType.T,
        coord_ref = maps.MapCoordReference.Equatorial)
    # Fill map with the point source mask
    sources.source_utils.make_point_source_map(
        ps_params, args.point_source_file)
    
    ps_map_id = 'PointSourceMask'
    filter_mask_key = 'FilterMask'
else:
    ps_map_id = None
    filter_mask_key = None
    
# =============================================================================
# Define useful functions and pipeline modules
# -----------------------------------------------------------------------------
def SimStub(fr, ts_key, valid_ids_key = 'valid_ids',
            flag_key='Flags', weight_key = 'TodWeights'):
    '''
    This function produces a separate .g3 file used later for making
    mock-observations. It records the active bolometers, detector weights,
    pointing, and flags in each scan frame.
    This output file is also useful for statistics on weights and flags.
    '''
    to_keep = [valid_ids_key, flag_key, weight_key,
               'RawBoresightAz', 'RawBoresightEl']
    for model in ['Offline','Online']:
        to_keep += [model+'BoresightAz', model+'BoresightEl',
                    model+'BoresightRa', model+'BoresightDec',
                    model+'RaDecRotation', model+'PointingModel']
   
    if fr.type == core.G3FrameType.Scan:
        if ts_key not in fr:
            raise KeyError("ts_key %s not present in scan frame!"%ts_key)
        fr[valid_ids_key] = core.G3VectorString(fr[ts_key].keys())
        for k in fr:
            if k not in to_keep:
                del fr[k]

def check_for_sim_keys(fr, valid_ids_key = 'valid_ids',
                       out_tsm_key = 'valid_ids_tsm'):
    '''
    CalculatePointing expects a TimestreamMap, so make sure there's one
    in the frame when doing mock-observations.
    valid_ids_key should already exist in simstub and point to a list.
    '''
    if fr.type == core.G3FrameType.Scan:
        if not valid_ids_key in fr:
            raise KeyError(
                "%s not found in frame. "%valid_ids_key +
                "List of valid bolometer ids required for mock-observing.")
        tsm = core.G3TimestreamMap()
        for bid in fr[valid_ids_key]:
            tsm[bid]=core.G3Timestream()
        fr[out_tsm_key] = tsm

def remove_missing_keys(fr, target_key, reference_key):
    '''
    Remove keys from fr[target_key] that are not in fr[reference_key].
    '''
    if not (target_key in fr and reference_key in fr):
        return
    old = fr.pop(target_key, None)
    new = type(old)()
    for k in old.keys():
        if k in fr[reference_key]:
            new[k] = old[k]
    fr[target_key] = new

def remove_present_in_list(fr, target_key, reference_list):
    '''
    Remove keys from fr[target_key] that are in reference_list.
    '''
    if not target_key in fr:
        return
    old = fr.pop(target_key, None)
    new = type(old)()
    for k in old.keys():
        if k not in reference_list:
            new[k] = old[k]
    fr[target_key] = new    
        
class ShowScanNumber(object):   
    def __init__(self):
        self.counter = 0
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Scan:
            print("\nInformation on Scan %s (0-indexed):\n"%self.counter)
            self.counter += 1

# =============================================================================
# Before pipeline, get list of wafers and bolos to exclude if required
# -----------------------------------------------------------------------------
bad_bolos = []
wafer_list = []
for fname in args.input_files:
    for frame in core.G3File(fname):
        if 'BolometerProperties' in frame:
            for bolo, props in frame['BolometerProperties'].items():
                if props.wafer_id not in [None,'']:
                    waf = props.wafer_id.capitalize()
                    if waf in bad_wafers:
                        bad_bolos.append(bolo)
                    elif len(good_wafers) > 0 and waf not in good_wafers:
                        bad_bolos.append(bolo)
                    elif waf not in wafer_list:
                        wafer_list.append(waf)
                if not np.isnan(props.band):
                    if (str(int(props.band/core.G3Units.GHz))+'GHz'
                        not in bands):
                        bad_bolos.append(bolo)
            break
    if len(wafer_list) > 0:
        break
bad_bolos = list(np.unique(bad_bolos))
args.wafers_to_include = wafer_list

# =============================================================================
# Begin pipeline
# -----------------------------------------------------------------------------
pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=args.input_files)

if not args.sim:
    # Drop timestreams from detectors that we're not interested in
    if len(bad_bolos) > 0:
        pipe.Add(remove_present_in_list, target_key = 'RawTimestreams_I',
                 reference_list = bad_bolos)
    raw_ts_key = 'RawTimestreams_I'
    raw_ts_key_i = 'RawTimestreams_I'
    raw_ts_key_q = 'RawTimestreams_Q'

    # -------------------------------------------------------------------------
    # Drop certain data before further processing
    # -------------------------------------------------------------------------
    # Cut turnarounds, deduplicate metadata
    pipe.Add(std_processing.DropWasteFrames)

    # Add scan numbers to scan frames
    pipe.Add(axiontod.AddScanNumberToFrame)

    # Drop scans depending on direction, if desired
    if isinstance(args.split_left_right, str):
        if args.split_left_right.lower().startswith('l'):
            pipe.Add(std_processing.ScanDirections.SelectScanDirection,
                     left = True)
        elif args.split_left_right.lower().startswith('r'):
            pipe.Add(std_processing.ScanDirections.SelectScanDirection,
                     left = False)

    # Remove anomalously short Scan frames
    pipe.Add(std_processing.ScanFlagging.RemoveShortScans)

    # Remove the occasional az wrap/unwrap
    if args.cut_az_unwraps:
        pipe.Add(std_processing.ScanFlagging.CutOnScanSpeed)

    # Remove frames containing an az glitch
    if args.cut_az_glitches:
        pipe.Add(std_processing.ScanFlagging.RemoveAzGlitches)

    # -------------------------------------------------------------------------
    # Flag detectors that were not operated properly and/or had invalid data
    # -------------------------------------------------------------------------
    pipe.Add(std_processing.flagsegments.FieldFlaggingPreKcmbConversion,
             flag_key = 'Flags', ts_key = raw_ts_key, flag_nan_offsets = True,
             flag_nan_bands = True)

    # Remove detectors from Flags that were never in data to begin with
    pipe.Add(remove_missing_keys,
             target_key = 'Flags', reference_key=raw_ts_key)
    # -------------------------------------------------------------------------
    # Calibrate timestreams
    # -------------------------------------------------------------------------
    pipe.Add(std_processing.CalibrateRawTimestreams,
            i_data_key = raw_ts_key, output = 'CalTimestreams')

    # Clean-up as we go
    pipe.Add(core.Delete, keys= [raw_ts_key_i, raw_ts_key_q, #'RawTimestreams_I', 'RawTimestreams_Q',
                                 'Notched_RawTimestreams'])

    # Collect statistics on flagged detectors
    stats = timestreamflagging.GenerateFlagStats(flag_key='Flags')
    pipe.Add(stats)

    # Remove flagged things before glitchfinding and other flagging
    # Removal after calibration so we don't have to remove the Q timestreams separately
    pipe.Add(timestreamflagging.RemoveFlagged,
             input_ts_key = 'CalTimestreams',
             input_flag_key = 'Flags',
             output_ts_key = 'DeflaggedCalTimestreams')
    pipe.Add(core.Delete, keys= ['CalTimestreams'])
    # -------------------------------------------------------------------------
    # Add pointing model
    # -------------------------------------------------------------------------
    # Delete pre-existing timestreams
    # Use this until we have a functional online pointing model.
    pipe.Add(core.Delete,
             keys=["OnlineBoresightAz",
                   "OnlineBoresightEl",
                   "OnlineBoresightRa",
                   "OnlineBoresightDec",
                   "OnlineRaDecRotation"])

    if args.pointing_model.lower().startswith('off'):
        pmodel = 'Offline'
        # Add Offline pointing to scans (if a model exists).
        pipe.Add(pointing.offline_pointing.ApplyPointingCorrection)

    elif args.pointing_model.lower().startswith('on'):
        pmodel = 'Online'

    pipe.Add(pointing.pointing_for_mapmaking.CalculateCoordTransRotations,
             raw_az_key = 'RawBoresightAz',
             raw_el_key = 'RawBoresightEl',
             output = pmodel+'Boresight',
             transform_store_key = pmodel+'RaDecRotation',
             model = pmodel+'PointingModel',
             flags = ['az_tilts', 'el_tilts',
                      'flexure', 'collimation', 'refraction'])
                  #, 'thermolin'] # Thermoline broken as of 4/13/17

    pointing_ts_key = 'DeflaggedCalTimestreams'

else:
    # Make sure a list of valid ids exists and create dummy timestream map
    pipe.Add(check_for_sim_keys, valid_ids_key = 'valid_ids',
             out_tsm_key = 'valid_ids_tsm')
    pipe.Add(axiontod.AddScanNumberToFrame)
    # Load the simulated map
    m = maps.fitsio.load_skymap_fits(args.sim_map)
    pipe.Add(maps.InjectMaps, map_id='sim_map',
             maps_in=[m['T'], m['Q'], m['U']])
    # Calculate sim_map pointing for simulated TOD
    if args.pointing_model.lower().startswith('off'):
        pmodel = 'Offline'
    elif args.pointing_model.lower().startswith('on'):
        pmodel = 'Online'
    pipe.Add(mapmaker.CalculatePointing,map_id = 'sim_map',
        pointing_store_key = 'SimPixelPointing',
        ts_map_key = 'valid_ids_tsm',
        trans_key = pmodel+'RaDecRotation')
    # Generate simulated TOD
    wk = 'TodWeights' if args.add_ts_noise else ''
    pipe.Add(mapmaker.mapmakerutils.FillSimTodSegment,
             out_ts_key = 'PreCMTimestreams',
             ts_to_get_sample_rate_key = pmodel+'BoresightRa',
             interp_sim = args.interp,
             trans_key = pmodel+'RaDecRotation',
             sim_map_id = 'sim_map',
             valid_ids = 'valid_ids',
             sim_pointing_key = 'SimPixelPointing',
             add_noise = args.add_ts_noise,
             mean = 0.0, weight_key = wk,
             seed = args.ts_noise_seed)
    pointing_ts_key = 'PreCMTimestreams'
    pre_poly_ts_key = 'PreCMTimestreams'

# -----------------------------------------------------------------------------
# Calculate pointing
# -----------------------------------------------------------------------------
pipe.Add(maps.InjectMaps, map_id='bsmap', maps_in=[map_params])

pipe.Add(mapmaker.mapmakerutils.CalculatePointing,
         map_id = 'bsmap',
         pointing_store_key = 'PixelPointing',
         trans_key = pmodel+'RaDecRotation',
         ts_map_key = pointing_ts_key)

# -----------------------------------------------------------------------------
# Add point source mask
# -----------------------------------------------------------------------------
if args.mask_point_sources and args.point_source_file not in [None, '']:
    pipe.Add(maps.InjectMaps, map_id=ps_map_id, maps_in=[ps_params])

if not args.sim:
    # -------------------------------------------------------------------------
    # More flagging and removal of flagged detectors
    # -------------------------------------------------------------------------
    pipe.Add(std_processing.flagsegments.FieldFlaggingPostKcmbConversion,
             flag_key = 'Flags', ts_key = 'DeflaggedCalTimestreams',
             filter_mask_key = filter_mask_key, ps_mask_id = ps_map_id,
             ps_pointing_key = 'PixelPointing')

    # Remove flagged things before common-mode filter
    pipe.Add(timestreamflagging.RemoveFlagged,
             input_ts_key = 'DeflaggedCalTimestreams',
             input_flag_key = 'Flags',
             output_ts_key = 'PreCMTimestreams')
    pre_poly_ts_key = 'PreCMTimestreams'

    pipe.Add(core.Delete, keys = ['DeflaggedCalTimestreams'])
    pipe.Add(stats.update)

    # If flagging has removed too many bolos, drop the frame.
    pipe.Add(std_processing.ScanFlagging.RemoveSparseScans,
             ts_key = 'PreCMTimestreams', min_num = args.minnum_bolos_per_scan)
    
# -----------------------------------------------------------------------------
# Common-mode filter
# -----------------------------------------------------------------------------
cm_mask_key = None
if args.apply_common_mode_filter:
    if args.mask_common_mode and ps_map_id is not None:
        cm_mask_key = 'FilterMask'
        pipe.Add(todfilter.FilterMaskInjector,
                 point_src_mask_id = ps_map_id,
                 filter_mask_key = cm_mask_key,
                 pointing_key = 'PixelPointing')

    pipe.Add(todfilter.commonmode.CommonModeFilter(
        in_ts_map_key = pre_poly_ts_key,
        out_ts_map_key = 'CMFilteredTimestreams',
        per_band = args.cm_by_band, per_wafer = args.cm_by_wafer,
        per_squid = args.cm_by_squid, filter_mask_key = cm_mask_key))
    pre_poly_ts_key = 'CMFilteredTimestreams'

# -----------------------------------------------------------------------------
# Timestream filtering
# -----------------------------------------------------------------------------

pipe.Add(mapmaker.TodFiltering,
         # filtering options
         poly_order = args.poly_order,
         filters_are_ell_based = args.filters_are_ell_based, 
         mhpf_cutoff = hpf,
         lpf_filter_frequency = lpf,
         point_source_mask_id = ps_map_id,
         # boiler plate
         ts_in_key = pre_poly_ts_key,
         ts_out_key = 'PolyFilteredTimestreams', 
         point_source_pointing_store_key = 'PixelPointing',
         use_dynamic_source_filter = False,
         boresight_az_key = pmodel+'BoresightAz',
         boresight_el_key = pmodel+'BoresightEl')

if args.apply_notch_filter:
    pipe.Add(todfilter.notchfilter.NotchFilterLLSVersion,
             input_tsm_key = 'PolyFilteredTimestreams',
             output_tsm_key = 'NotchFilteredTimestreams',
             use_masks = args.mask_point_sources)
    pipe.Add(core.Delete, keys = ['PolyFilteredTimestreams'])
    post_filtering_tsm_key = 'NotchFilteredTimestreams'
else:
    post_filtering_tsm_key = 'PolyFilteredTimestreams'

# More clean-up
pipe.Add(core.Delete, keys=['CMFilteredTimestreams'])

if not args.sim:
    # -------------------------------------------------------------------------
    # Calculate weights, flag bolos with bad weights
    # -------------------------------------------------------------------------
    pipe.Add(std_processing.weighting.AddPSDWeights,
             input = post_filtering_tsm_key, output = 'TodWeights',
             low_f = args.weight_low_freq * core.G3Units.Hz,
             high_f = args.weight_high_freq * core.G3Units.Hz)

    pipe.Add(timestreamflagging.flaggingutils.SigmaclipFlagGroupG3MapValue,
             m_key = 'TodWeights', low = 3, high = 3, per_band = True,
             flag_reason = 'BadWeight', flag_key = 'Flags')


# -----------------------------------------------------------------------------
# Apply the scan flagging 
# -----------------------------------------------------------------------------
if args.equal_flags_in_scan_pairs:
    pipe.Add(axiontod.ForceEqualBoloFlagsForScanPairs, flag_key='Flags')


# -----------------------------------------------------------------------------
# Remove flagged bolos
# -----------------------------------------------------------------------------
pipe.Add(timestreamflagging.RemoveFlagged, 
         input_ts_key = post_filtering_tsm_key,
         input_flag_key = 'Flags',
         output_ts_key = 'DeflaggedTimestreams')
pipe.Add(core.Delete, keys=['PolyFilteredTimestreams',
                            'NotchFilteredTimestreams'])
deflagged_ts_key = 'DeflaggedTimestreams'


# -----------------------------------------------------------------------------
# Multiply scans by +/- in order to cancel CMB and create a noise simulation
# -----------------------------------------------------------------------------
# if args.sign_flip_noise:
#     pipe.Add(axiontod.FlipScanSigns(seed=0,
#                                     input_key_i=deflagged_ts_key,
#                                     input_key_q=None,
#                                     output_key_i=deflagged_ts_key + 'Flipped',
#                                     output_key_q=None,
#                                     az_key=pmodel+'BoresightAz',
#                                     random=False))
#     pipe.Add(core.Delete, keys=[deflagged_ts_key])
#     deflagged_ts_key = deflagged_ts_key + 'Flipped'

#     if args.verbose:
#         core.set_log_level(core.G3LogLevel.LOG_DEBUG, 'FlipScanSigns')

# -----------------------------------------------------------------------------
# Generate map meta-data to access later
# -----------------------------------------------------------------------------
if args.sim:
    stats = timestreamflagging.GenerateFlagStats(flag_key='Flags')
    pipe.Add(stats)
else:
    pipe.Add(stats.update)

pipe.Add(std_processing.AddMetaData, stats = stats, args = vars(args))


# -----------------------------------------------------------------------------
# Bin the timestreams into maps
# -----------------------------------------------------------------------------
if args.split_left_right:
    if str(args.split_left_right).lower().startswith('l'):
        directions=['Left']
    elif str(args.split_left_right).lower().startswith('r'):
        directions=['Right']
    else:
        directions = ['Left', 'Right']
    pipe.Add(std_processing.ScanDirections.split_left_right_scans,
             ts_in_key = deflagged_ts_key,
             boresight_az_key = pmodel+'BoresightAz')
    pipe.Add(core.Delete, keys=[deflagged_ts_key])
else:
    directions = ['']

for direction in directions:
    if args.split_by_band:
        pipe.Add(calibration.SplitByBand,
                 input = deflagged_ts_key+direction,
                 output_root = deflagged_ts_key+direction)
        pipe.Add(core.Delete, keys=[deflagged_ts_key+direction])
            
        if args.split_by_wafer:
            wafers = wafer_list
            for band in bands:
                pipe.Add(calibration.SplitByWafer,
                         input = deflagged_ts_key+direction+band,
                         output_root = deflagged_ts_key+direction+band)
                pipe.Add(core.Delete,
                         keys=[deflagged_ts_key+direction+band])
        else:
            wafers = ['']

    elif args.split_by_wafer:
        bands = ['']
        wafers = wafer_list
        pipe.Add(calibration.SplitByWafer,
                 input = deflagged_ts_key+direction,
                 output_root = deflagged_ts_key+direction)
        pipe.Add(core.Delete, keys=[deflagged_ts_key+direction])

    else:
        bands, wafers = [''], ['']

if args.verbose:
    pipe.Add(ShowScanNumber)
    pipe.Add(core.Dump)

mapids = ['%s%s%s' % (direction, band, wafer)
          for direction in directions for band in bands for wafer in wafers]

if args.sign_flip_noise:
    def emit_map(frame):
        if frame.type == core.G3FrameType.Scan:
            frame['EmitMap'] = core.G3Int(True)

    if args.avg_lr_weights:
        pipe.Add(axiontod.AvgTodWeights, weight_key='TodWeights')
    pipe.Add(emit_map)

for mapid in mapids:
    if args.sign_flip_noise:
        pipe.Add(axiontod.InjectMapStubPerScan,
                 map_id=mapid,
                 map_stub=map_params,
                 polarized=(not args.temperature_only),
                 weighted=True)
    else:
        pipe.Add(maps.InjectMapStub,
                 map_id=mapid,
                 map_stub=map_params,
                 polarized=(not args.temperature_only),
                 weighted=True)

    pipe.Add(mapmaker.mapmakerutils.BinMap,
                 map_id=mapid,
                 ts_map_key = deflagged_ts_key + mapid,
                 trans_key = pmodel+'RaDecRotation',
                 pointing_store_key = 'PixelPointing',
                 timestream_weight_key = 'TodWeights')


# -----------------------------------------------------------------------------
# Write simstub
# -----------------------------------------------------------------------------
if args.produce_simstub:
    pipe.Add(SimStub, ts_key = 'PreCMTimestreams')
    pipe.Add(core.G3Writer,
             filename = args.output.replace(
                 args.output.split('/')[-1],
                 'simstub_'+args.output.split('/')[-1]),
             streams=[core.G3FrameType.Observation, core.G3FrameType.Wiring,
                      core.G3FrameType.Calibration, core.G3FrameType.Scan,
                      core.G3FrameType.PipelineInfo,
                      core.G3FrameType.EndProcessing])

# -----------------------------------------------------------------------------
# Drop Scan frames and maps without sky data
# -----------------------------------------------------------------------------
pipe.Add(lambda fr: fr.type != core.G3FrameType.Scan)
pipe.Add(lambda fr: 'Id' not in fr or fr['Id'] != 'bsmap')
pipe.Add(lambda fr: 'Id' not in fr or fr['Id'] != 'PointSourceMask')

# -----------------------------------------------------------------------------
# Write to file
# -----------------------------------------------------------------------------
pipe.Add(core.G3Writer, filename = args.output)

# -----------------------------------------------------------------------------
# Write Temperature-only maps to file
# -----------------------------------------------------------------------------
if not args.temperature_only and args.add_temperature_only:
    pipe.Add(maps.map_modules.MakeMapsUnpolarized)
    pipe.Add(core.G3Writer, filename = args.output.replace('.g3', '_tonly.g3'))

# =============================================================================
# Run the pipeline
# -----------------------------------------------------------------------------
pipe.Run(profile=True)
