from spt3g import core, calibration
import matplotlib.pyplot as plt
import pydfmux
import numpy as np

d = list(core.G3File('/home/adama/SPT/spt3g_software/calibration/scripts/new_boloprops.g3'))[0] #'/spt/user/production/calibration/calframe/RCW38-pixelraster/70228757.g3'))[0]
dnominal = list(core.G3File('/spt/data/bolodata/downsampled/RCW38-pixelraster/70228757/nominal_online_cal.g3'))[0]
hwm = pydfmux.load_session(open('/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'))['hardware_map']

x = {}
y = {}
x_nominal = {}
y_nominal = {}
x_hwm = {}
y_hwm = {}

for bolo in d['BolometerProperties'].keys():
    if d['BolometerProperties'][bolo].wafer_id not in x:
        x[d['BolometerProperties'][bolo].wafer_id] = {}
        y[d['BolometerProperties'][bolo].wafer_id] = {}
    x[d['BolometerProperties'][bolo].wafer_id][bolo] = d['BolometerProperties'][bolo].x_offset
    y[d['BolometerProperties'][bolo].wafer_id][bolo] = d['BolometerProperties'][bolo].y_offset

for bolo in dnominal['NominalBolometerProperties'].keys():
    if dnominal['NominalBolometerProperties'][bolo].wafer_id not in x_nominal:
        x_nominal[dnominal['NominalBolometerProperties'][bolo].wafer_id] = {}
        y_nominal[dnominal['NominalBolometerProperties'][bolo].wafer_id] = {}
    x_nominal[dnominal['NominalBolometerProperties'][bolo].wafer_id][bolo] = dnominal['NominalBolometerProperties'][bolo].x_offset
    y_nominal[dnominal['NominalBolometerProperties'][bolo].wafer_id][bolo] = dnominal['NominalBolometerProperties'][bolo].y_offset
    
bololist = hwm.query(pydfmux.Bolometer)
for bolo in bololist:
    if bolo.wafer.name not in x_hwm:
        x_hwm[bolo.wafer.name] = {}
        y_hwm[bolo.wafer.name] = {}
    x_hwm[bolo.wafer.name][bolo.name] = bolo.x_mm
    y_hwm[bolo.wafer.name][bolo.name] = bolo.y_mm

bololist_bps = list(d['BolometerProperties'].keys())
ndifferent = 0
for bolo in bololist:
    if bolo.name not in bololist_bps:
        print('Bolometer {} is not in BolometerProperties!'.format(bolo.name))
    elif int(d['BolometerProperties'][bolo.name].pixel_id) != int(bolo.pixel):
        print('Bolometer {} has different matching between HWM and boloprops!'.format(bolo.name))
        print(d['BolometerProperties'][bolo.name].pixel_id)
        print(bolo.pixel)
        ndifferent += 1
print('Total number of detectors with different pixels numbers between HWM and boloprops = {}'.format(ndifferent))
        
plt.figure()
print(x.keys())
for wafer in x:
    x_plot = [x[wafer][bolo] for bolo in x[wafer].keys()]
    y_plot = [y[wafer][bolo] for bolo in y[wafer].keys()]

    plt.plot(x_plot, y_plot, '.', label=wafer)
plt.axis([-0.02, 0.02, -0.02, 0.02])
plt.savefig('wafertest.png', dpi=200)

plt.figure()
for wafer in x:
    if wafer != '':
        x_nominal_common = [x_nominal[wafer][bolo] for bolo in x_nominal[wafer].keys()]
        y_nominal_common = [y_nominal[wafer][bolo] for bolo in y_nominal[wafer].keys()]

    plt.plot(x_nominal_common, y_nominal_common, '.', label=wafer)
plt.axis([-0.02, 0.02, -0.02, 0.02])
plt.savefig('wafertest_nominal.png', dpi=200)

plt.figure()
for wafer in x:
    if wafer != '':
        plt.plot(x_hwm[wafer].values(),
                 y_hwm[wafer].values(), '.', label=wafer)
plt.axis([-200, 200, -200, 200])
plt.savefig('wafertest_hwm.png', dpi=200)

plt.figure()
for wafer in x:
    bolos_common = np.intersect1d(list(x_hwm[wafer].keys()),
                                 list(x_nominal[wafer].keys()))
    x_hwm_common = [x_hwm[wafer][bolo] for bolo in bolos_common]
    y_hwm_common = [y_hwm[wafer][bolo] for bolo in bolos_common]
    x_nominal_common = [x_nominal[wafer][bolo] for bolo in bolos_common]
    y_nominal_common = [y_nominal[wafer][bolo] for bolo in bolos_common]
    plt.plot(x_hwm_common, x_nominal_common, '.', label=wafer)
plt.axis([-200,200,-0.02,0.02])
plt.legend()
plt.savefig('x_hwmVnominal.png', dpi=200)


plt.figure()
for wafer in x:
    bolos_common = np.intersect1d(list(x[wafer].keys()),
                                  list(x_nominal[wafer].keys()))
    x_common = np.array([x[wafer][bolo] for bolo in bolos_common])
    y_common = np.array([y[wafer][bolo] for bolo in bolos_common])
    x_nominal_common = np.array([x_nominal[wafer][bolo] for bolo in bolos_common])
    y_nominal_common = np.array([y_nominal[wafer][bolo] for bolo in bolos_common])

    delta_r = np.sqrt((x_common - x_nominal_common)**2. + \
                      (y_common - y_nominal_common)**2.)
    plt.hist(delta_r, bins=np.linspace(0,0.004,51), histtype='step')
    print(bolos_common[delta_r>0.003])

plt.gca().set_yscale('log')
plt.legend()
plt.savefig('deltar_rcw38Vnominal.png', dpi=200)
