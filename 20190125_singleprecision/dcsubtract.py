from spt3g import core
import numpy as np

def subtract_dc_offset(frame, ts_key_in='RawTimestreams_I',
                       ts_key_out='DCSubtractTimestreams'):
    if frame.type == core.G3FrameType.Scan and \
       ts_key_in in frame.keys():
        if ts_key_out not in frame.keys():
            frame[ts_key_out] = core.G3TimestreamMap()
        for bolo in frame[ts_key_in].keys():
            frame[ts_key_out][bolo] = frame[ts_key_in][bolo] - np.round(np.mean(frame[ts_key_in][bolo]))
