from spt3g import core
from spt3g.timestreamflagging.miscflagmodules import FlagIncompletePixelPairs

pipe = core.G3Pipeline()
pipe.Add(core.G3Reader, filename=['/spt/data/bolodata/fullrate/calibrator/70005920/offline_calibration.g3',
                                  '/spt/data/bolodata/fullrate/calibrator/70005920/0000.g3'])
pipe.Add(FlagIncompletePixelPairs, ts_key='RawTimestreams_I')
pipe.Add(FlagIncompletePixelPairs, ts_key='RawTimestreams_I')
pipe.Add(core.G3Writer, filename='flagged_pairs.g3')
pipe.Run()

pipe.Add(core.G3Reader, filename='flagged_pairs.g3')
pipe.Add(core.Dump)
pipe.Run()
