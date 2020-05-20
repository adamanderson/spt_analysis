from spt3g import core, calibration, dfmux, mapmaker
import numpy as np

fsingle = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'singleprecision/64502043_tod_v13.g3')
fdouble = core.G3File('/home/adama/SPT/spt_analysis/20190125_singleprecision/' + \
                      'doubleprecision/64502043_tod_v13.g3')
write_mix = core.G3Writer('mixedprecision/64502043_v13_single_pixelpointing.g3')

fr_single = fsingle.next()
fr_double = fdouble.next()
while True:
    if fr_single.type == core.G3FrameType.Scan and \
       fr_double.type == core.G3FrameType.Scan:
        new_frame = core.G3Frame(core.G3FrameType.Scan)
        new_frame['DeflaggedTimestreams'] = fr_double['DeflaggedTimestreams']
        new_frame['OnlineRaDecRotation'] = fr_double['OnlineRaDecRotation']
        new_frame['PixelPointing'] = fr_single['PixelPointing']
        new_frame['TodWeights'] = fr_double['TodWeights']

        bolos_double = np.array(list(fr_double['DeflaggedTimestreams'].keys()))
        bolos_pointing = np.array(list(fr_single['PixelPointing'].keys()))
        bolos_toremove = np.setdiff1d(bolos_double, bolos_pointing)

        for bolo in bolos_toremove:
            if bolo in new_frame['DeflaggedTimestreams']:
                new_frame['DeflaggedTimestreams'].pop(bolo)

        print(new_frame)
        write_mix.Process(new_frame)
    else:
        write_mix.Process(fr_double)

    try:
        fr_single = fsingle.next()
        fr_double = fdouble.next()
    except:
        break

write_mix.Process(core.G3Frame(core.G3FrameType.EndProcessing))
