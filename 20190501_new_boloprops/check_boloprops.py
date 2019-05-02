from spt3g import core, calibration
import matplotlib.pyplot as plt


d = list(core.G3File('/spt/user/production/calibration/calframe/RCW38-pixelraster/64599638.g3'))[0]

x = {}
y = {}
for bolo in d['BolometerProperties'].keys():
    if d['BolometerProperties'][bolo].wafer_id not in x:
        x[d['BolometerProperties'][bolo].wafer_id] = []
        y[d['BolometerProperties'][bolo].wafer_id] = []
    
    x[d['BolometerProperties'][bolo].wafer_id].append(d['BolometerProperties'][bolo].x_offset)
    y[d['BolometerProperties'][bolo].wafer_id].append(d['BolometerProperties'][bolo].y_offset)


plt.figure()
for wafer in x:
    plt.plot(x[wafer], y[wafer], '.', label=wafer)
plt.savefig('wafertest.png', dpi=200)
