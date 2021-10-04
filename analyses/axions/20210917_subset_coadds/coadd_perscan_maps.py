from spt3g import core, maps, util
from glob import glob
import argparse, pickle as pk
from spt3g.std_processing import time_to_obsid
from spt3g import util
import axiontod
import os
import numpy as np

class SkipMaxScan(object):
    def __init__(self):
        self.max_scan_num = None
    
    def __call__(self, frame):
        if frame.type == core.G3FrameType.Map:
            if self.max_scan_num is None or frame['ScanNumber'] > self.max_scan_num:
                self.max_scan_num = frame['ScanNumber']
            else:
                return []

def fname_print(frame):
    if frame.type != core.G3FrameType.PipelineInfo:
        return                             
    if 'output' in frame:                                               
        print(frame['output'])

map_dir = '/sptgrid/user/kferguson/axion_perscan_maps_2019_every_scan'
field = 'ra0hdec-59.75'
band = 90
fnames = np.sort(glob(os.path.join(map_dir, '{}_{}GHz_*.g3.gz'.format(field, band))))
obsids = []
for fname in fnames:
    obsids.append(int(os.path.splitext(os.path.splitext(os.path.basename(fname))[0])[0].split('_')[-1]))
obsids = np.array(obsids)
month_list = np.arange(3,12)
time_edges = [time_to_obsid('2019{:02}01_000000'.format(month)) for month in month_list]


for jtime in range(len(time_edges) - 1):
    pipe = core.G3Pipeline()
    pipe.Add(core.G3Reader, filename=fnames[(obsids>time_edges[jtime]) & (obsids<time_edges[jtime+1])])
    pipe.Add(SkipMaxScan)
    pipe.Add(core.Dump)
    pipe.Add(util.framecombiner.MapFrameCombiner, fr_id = '*%sGHz'%band)
    # pipe.Add(axiontod.MapCombiner, fr_id = '*%sGHz'%band)
    pipe.Add(core.G3Writer,
             filename='coadd_{}_{}GHz_2019{:02d}_to_2019{:02d}.g3.gz'.format(field, band,
                                                                             month_list[jtime],
                                                                             month_list[jtime]+1),
             streams=[core.G3FrameType.Map])
    pipe.Run()

