'''
Script to make CMB field maps.
'''
import argparse
from spt3g import core, std_processing, mapmaker, calibration
from spt3g import timestreamflagging, todfilter, coordinateutils
from spt3g.std_processing.pointing import CalculateCoordTransRotations
from copy import deepcopy
# Usage: field_autoproc.py <input files.g3> -o outputmaps.g3
parser = argparse.ArgumentParser(description='Maps for a CMB field')
parser.add_argument('input_files', nargs = '+')
parser.add_argument('-o', '--output', default = 'output.g3',
                   help='Output filename')
parser.add_argument('-s', '--source', default = 'ra0hdec-57.5',
                   help='Name of source, to set field center')
parser.add_argument('-r', '--res', default = 2.0, type=float,
                   help='resolution [arcmin]')
parser.add_argument('-x', '--xlen', default = 75, type=float,
                   help='map width [deg]')
parser.add_argument('-y', '--ylen', default = 50, type=float,
                   help='map height [deg]')
parser.add_argument('--psfile', default = None,
                   help = 'Point source configuration file for making a point source mask')
parser.add_argument('-v', '--verbose', action = 'store_true', default=False,
                    help = 'Print every frame')
parser.add_argument('--lr', action = 'store_true',default=False,
                    help = 'Split left-right')
parser.add_argument('--tonly', default = True, action = 'store_false',
                    help = 'Include this flag to make T-only maps')
parser.add_argument('--simstub', default = False, action = 'store_true',
                    help = 'Include this flag to produce a simstub')
args = parser.parse_args()

args.res *= core.G3Units.arcmin
args.xlen *= core.G3Units.deg
args.ylen *= core.G3Units.deg
x_len = int(args.xlen / args.res)
y_len = int(args.ylen / args.res)

# Suppress all warnings about timestream missing from scan frame
core.set_log_level(core.G3LogLevel.LOG_ERROR, 'MapBinner')

def SimStub(fr, ts_key, valid_ids_key = 'valid_ids',
            flag_key='Flags', weight_key = 'TodWeights'):
    
    to_keep = [valid_ids_key, flag_key, weight_key,
               'DfMuxHousekeeping',
               'RawBoresightAz', 'RawBoresightEl',
               'OnlineBoresightAz','OnlineBoresightEl',
               'OnlineBoresightRa','OnlineBoresightDec',
               'OnlineRaDecRotation','OnlinePointingModel']
   
    if fr.type == core.G3FrameType.Scan:
        assert(ts_key in fr)
        fr[valid_ids_key] = core.G3VectorString(fr[ts_key].keys())
        for k in fr:
            if k not in to_keep:
                del fr[k]
                
# Generate map stub
map_params = std_processing.CreateSourceMapStub(
    args.source, x_len = x_len, y_len = y_len, res = args.res,
    proj = coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea)

if args.psfile is not None:
    # Set up an empty map for point source filtering
    ps_params = std_processing.CreateSourceMapStub(
        args.source, x_len = x_len, y_len = y_len, res = args.res,
        proj = coordinateutils.MapProjection.ProjLambertAzimuthalEqualArea)
    # Now, fill map with the point source mask
    mapmaker.pointsourceutils.make_point_source_map(ps_params, args.psfile)
    
    ps_map_id = 'PointSourceMask'
else:
    ps_map_id = None

# Begin pipeline
pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=args.input_files)

#### TEST
# Write to file
pipe.Add(core.Dump)
pipe.Add(core.G3Writer, filename = args.output)
pipe.Run(profile=True)
#### TEST


# # Cut turnarounds, deduplicate metadata
# pipe.Add(std_processing.DropWasteFrames)

# # Remove the occasional az wrap/unwrap
# pipe.Add(std_processing.ScanFlagging.CutOnScanSpeed)

# # Drop bolos on w201, they have too many lines
# class RemoveW201(object):
#     def __init__(self, input = 'RawTimestreams_I'):
#         self.input = input
#         self.bp = None
#         self.bad_bolos=[]
#     def __call__(self, fr):
#         if fr.type == core.G3FrameType.Calibration:
#             self.bp = fr['BolometerProperties']
#         if self.input in fr:
#             if len(self.bad_bolos) == 0:
#                 for bolo in fr[self.input].keys():
#                     if bolo not in self.bp.keys() or self.bp[bolo].wafer_id == 'w201':
#                         fr[self.input].pop(bolo, None)
#                         self.bad_bolos.append(bolo)
#             else:
#                 for bolo in self.bad_bolos:
#                     fr[self.input].pop(bolo, None)
#         else:
#             return
# pipe.Add(RemoveW201)

# # Flag junk
# pipe.Add(std_processing.flagsegments.FieldFlaggingPreKcmbConversion,
#          flag_key = 'Flags', ts_key = 'RawTimestreams_I')

# # Apply calibrations
# pipe.Add(std_processing.CalibrateRawTimestreams,
#         output = 'CalTimestreams')

# # More flagging
# pipe.Add(std_processing.flagsegments.FieldFlaggingPostKcmbConversion,
#          flag_key = 'Flags', ts_key = 'CalTimestreams')

# def Keep(fr, ts_key, filter_key):
#     # Remove bolos from 'Flags' that are not in ts_key.
#     # This is just to clean up Flag stats
#     if not (ts_key in fr and filter_key in fr):
#         return
#     old = fr.pop(ts_key,None)
#     new = type(old)()
#     for k in old.keys():
#         if k in fr[filter_key]:
#             new[k] = old[k]
#     fr[ts_key] = new
# pipe.Add(Keep, ts_key = 'Flags', filter_key='CalTimestreams')

# # Remove flagged things before common-mode filter
# pipe.Add(timestreamflagging.RemoveFlagged,
#          input_ts_key = 'CalTimestreams',
#          input_flag_key = 'Flags',
#          output_ts_key = 'PreCMTimestreams')

# # If flagging has removed too many bolos, drop the frame.
# def ts_check(fr, ts_key):
#     if fr.type == core.G3FrameType.Scan:
#         if ts_key not in fr or len(fr[ts_key].keys())<1000:
#             return False
# pipe.Add(ts_check, ts_key = 'PreCMTimestreams')

# # Common-mode filter
# pipe.Add(todfilter.polyutils.CommonModeFilter(
#     in_ts_map_key = 'CalTimestreams',
#     out_ts_map_key = 'CMFilteredTimestreams',
#     per_band = True, per_wafer = True, per_squid = False))

# # Clean-up as we go
# pipe.Add(core.Delete, keys= ['RawTimestreams_I','CalTimestreams'])

        
# # Add point source mask and calculate detector pointing
# if args.psfile is not None:
#     pipe.Add(mapmaker.MapInjector, map_id =  ps_map_id,
#              maps_lst = [ps_params,], is_stub=False)   
# pipe.Add(mapmaker.MapInjector, map_id = 'bsmap',
#          maps_lst = [map_params,], is_stub=False)

# Add Offline pointing to scans (if a model exists).
# pipe.Add(
#     CalculateCoordTransRotations,
#     raw_az_key='RawBoresightAz',
#     raw_el_key='RawBoresightEl',
#     output='OfflineBoresight',
#     transform_store_key='OfflineRaDecRotation',
#     model='OfflinePointingModel',
#     flags=['az_tilts', 'el_tilts', 'flexure', 'collimation', 'refraction']
#     #, 'thermolin'] # Thermoline broken as of 4/13/17
# )

# Add Online pointing to scans (use this until we have an online pointing model).
# Clean up pre-existing timestreams
# pipe.Add(
#     core.Delete,
#     keys=['OnlineBoresightAz', 'OnlineBoresightEl',
#           'OnlineBoresightRa', 'OnlineBoresightDec', 'OnlineRaDecRotation']
# )
# pipe.Add(
#     CalculateCoordTransRotations,
#     raw_az_key='RawBoresightAz',
#     raw_el_key='RawBoresightEl',
#     output='OnlineBoresight',
#     transform_store_key='OnlineRaDecRotation',
#     model='OnlinePointingModel',
#     flags=['az_tilts', 'el_tilts', 'flexure', 'collimation', 'refraction']
#     #, 'thermolin'] # Thermoline broken as of 4/13/17
# )

# pipe.Add(mapmaker.mapmakerutils.CalculatePointing, 
#          map_id = 'bsmap', 
#          pointing_store_key = 'PixelPointing', trans_key='OnlineRaDecRotation',
#          ts_map_key = 'PreCMTimestreams')

# # Timestream filtering
# pipe.Add(mapmaker.TodFiltering,
#          # filtering options
#          # poly_order = 19,
#          # filters_are_ell_based = True, 
#          # mhpf_cutoff = 300, lpf_filter_frequency = 6600,
#          # point_source_mask_id = ps_map_id,
#          # boiler plate
#          ts_in_key='PreCMTimestreams',
#          ts_out_key = 'PolyFilteredTimestreams', 
#          point_source_pointing_store_key = 'PixelPointing',
#          use_dynamic_source_filter = False,
#          boresight_az_key='OnlineBoresightAz',
#          boresight_el_key='OnlineBoresightEl')

# # Calculate Weights
# pipe.Add(std_processing.weighting.AddPSDWeights,
#          input = 'PolyFilteredTimestreams', output = 'TodWeights',
#          low_f = 0.75 * core.G3Units.Hz,
#          high_f = 8 * core.G3Units.Hz)

# # Flag bolos with bad weights
# pipe.Add(timestreamflagging.flaggingutils.SigmaclipFlagGroupG3MapValue,
#          m_key = 'TodWeights', low = 3, high = 3, per_band = True,
#          flag_reason = 'BadWeight', flag_key = 'Flags')

# pipe.Add(timestreamflagging.GenerateFlagStats, flag_key = 'Flags')
# pipe.Add(timestreamflagging.RemoveFlagged, 
#          input_ts_key = 'PolyFilteredTimestreams',
#          input_flag_key = 'Flags',
#          output_ts_key = 'DeflaggedTimestreams')

# # More clean-up
# pipe.Add(core.Delete, keys=['PolyFilteredTimestreams','PreCMTimestreams'])

# if args.lr:
#     # split left and right scans if requested
#     pipe.Add(std_processing.pointing.split_left_right_scans)
#     # Split by band (could do other things: wafer, bolos, etc.)
#     for direction in ['Left', 'Right']:
#         pipe.Add(calibration.SplitByBand,
#                  input='DeflaggedTimestreams' + direction,
#                  output_root='DeflaggedTimestreams' + direction)
#     if args.verbose:
#         pipe.Add(core.Dump)
#     for direction in ['Left', 'Right']:
#         for band in ['90', '150', '220']: # XXX should be automatic
#             mapid = '%s-%sGHz' %(direction, band)
#             pipe.Add(mapmaker.MapInjector, map_id = mapid,
#                      maps_lst=[map_params], is_stub=True, 
#                      make_polarized=args.tonly, do_weight=True)
#             pipe.Add(mapmaker.mapmakerutils.BinMap, map_id = mapid,
#                      ts_map_key='DeflaggedTimestreams%s%sGHz' %(direction, 
#                                                                  band),
#                      trans_key='OnlineRaDecRotation',
#                      pointing_store_key='PixelPointing', 
#                      timestream_weight_key = 'TodWeights')
# else:
#     # Split by band (could do other things: wafer, bolos, etc.)
#     pipe.Add(calibration.SplitByBand,
#              input='DeflaggedTimestreams',
#              output_root='DeflaggedTimestreams')
#     if args.verbose:
#         pipe.Add(core.Dump)
#     # Kick off maps
#     for band in ['90', '150', '220']: # XXX should be automatic
#         mapid = '%sGHz' % band
#         pipe.Add(mapmaker.MapInjector, map_id = mapid,
#                  maps_lst=[map_params], is_stub=True, 
#                  make_polarized=args.tonly, do_weight=True)
#         pipe.Add(mapmaker.mapmakerutils.BinMap, map_id = mapid,
#                  ts_map_key='DeflaggedTimestreams%sGHz' % band,
#                  trans_key='OnlineRaDecRotation',
#                  pointing_store_key='PixelPointing', 
#                  timestream_weight_key = 'TodWeights')

# if args.simstub:
#     # Write sim stub
#     pipe.Add(SimStub, ts_key = 'PreCMTimestreams')
#     pipe.Add(core.G3Writer,
#              filename = args.output.replace(args.output.split('/')[-1],
#                                             'simstub_'+args.output.split('/')[-1]),
#              streams=[core.G3FrameType.Observation, core.G3FrameType.Wiring,
#                       core.G3FrameType.Calibration, core.G3FrameType.Scan,
#                       core.G3FrameType.EndProcessing])
# # Drop TOD    
# pipe.Add(lambda fr: fr.type != core.G3FrameType.Scan)

