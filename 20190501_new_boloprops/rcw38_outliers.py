from spt3g import core, calibration
import numpy as np
import scipy
import matplotlib.pyplot as plt

def robust_avg(data):
                gooddata = np.asarray(data)[np.isfinite(data)]
                if len(gooddata) == 1:
                    return gooddata[0]
                return np.median(scipy.stats.sigmaclip(gooddata,
                                                          low=2.0, high=2.0)[0])

dcal = list(core.G3File('60000000.g3'))[0]
xoffsets = {}
yoffsets = {}
for fname in dcal["BolometerPropertiesBasedOn"]:
    if 'nominal_online_cal' not in fname:
        d = list(core.G3File(fname))[0]
        for bolo in d['PointingOffsetX'].keys():
            if bolo not in xoffsets:
                xoffsets[bolo] = []
                yoffsets[bolo] = []
            xoffsets[bolo].append(d['PointingOffsetX'][bolo])
            yoffsets[bolo].append(d['PointingOffsetY'][bolo])

xoffsets_avg = {}
yoffsets_avg = {}
for bolo in xoffsets:
    xoffsets_avg[bolo] = robust_avg(xoffsets[bolo])
    yoffsets_avg[bolo] = robust_avg(yoffsets[bolo])


xplot = np.array([xoffsets_avg[bolo] for bolo in xoffsets_avg])
yplot = np.array([yoffsets_avg[bolo] for bolo in yoffsets_avg])
plt.figure()
plt.plot(xplot, yplot, '.')
plt.savefig('rcw38_xVy.png', dpi=200)
