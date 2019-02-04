import pydfmux
import pandas as pd
from pydfmux.spt3g.hwm_tools import wafer_bolo_info
from pydfmux.spt3g import lc_tools
import numpy as np
import pickle

hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
hwm = pydfmux.load_session(open(hwm_fname, 'rb'))['hardware_map']
bolos = hwm.query(pydfmux.Bolometer)

# load the matching info
df_swap = pd.read_csv('singleton_fixes.csv', sep='\t')
df_info = wafer_bolo_info()
f_new = {}
for ind, row in df_swap.iterrows():
    old_bolo = df_swap['old_name'][ind].split('_')[1]
    old_wafer = df_swap['old_name'][ind].split('_')[0]
    new_bolo = df_swap['new_name'][ind].split('_')[1]
    new_wafer = df_swap['new_name'][ind].split('_')[0]
    bolo = hwm.query(pydfmux.Bolometer).join(pydfmux.Wafer)\
                                       .filter((pydfmux.Bolometer.physical_name==old_bolo) & \
                                               (pydfmux.Wafer.name == old_wafer))
    old_lc_ind = bolo[0].channel_map.lc_channel.channel-1
    new_lc_ind = np.array(df_info[df_info.physical_name==new_bolo].lc_ind)[0]
    lc_name = bolo[0].channel_map.lc_channel.lc_board.name
    version = lc_name.split('.')[1]
    batch = lc_name.split('.')[2]

    print(lc_name)
    print('{} -> {}'.format(old_lc_ind, new_lc_ind))

    if lc_name not in f_new:
        if lc_name in lc_tools.template_data[version].keys():
            f_new[lc_name] = lc_tools.f_expected[version][lc_name]
        else:
            f_new[lc_name] = lc_tools.f_expected[version][batch]
    old_template = f_new[lc_name].copy()

    f_new[lc_name][new_lc_ind] = bolo[0].channel_map.lc_channel.frequency

    # check that frequencies are still monotonic; we often bump one frequency
    # above or below its neighbor, which can result in badness
    if new_lc_ind < 67 and \
       f_new[lc_name][new_lc_ind] > f_new[lc_name][new_lc_ind+1]:
        f_new[lc_name][new_lc_ind+1] = f_new[lc_name][new_lc_ind] + 2000
    if new_lc_ind > 0 and \
       f_new[lc_name][new_lc_ind] < f_new[lc_name][new_lc_ind-1]:
        f_new[lc_name][new_lc_ind-1] = f_new[lc_name][new_lc_ind] - 2000

    # double-check that monotonicity has been restored
    if not np.all(np.sort(f_new[lc_name]) == f_new[lc_name]):
        print('Template for {} is still not monotonic despite adjusting '
              'nearest-neighbor frequencies.'.format(lc_name))
        print(old_template)
        print(np.sort(f_new[lc_name]))
        print(f_new[lc_name])
        print(np.sort(f_new[lc_name]) - f_new[lc_name])

for lc_name in f_new:
    with open('{}_custom_template.pkl'.format(lc_name.lower()), 'wb') as f:
        pickle.dump(f_new[lc_name], f, protocol=2, fix_imports=True)
