import pydfmux
import numpy as np
import pickle

load_hwm = True

# some checks of the hardware map after making changes to the bolometer mapping

# load data from the hardware maps
if load_hwm:
    old_hwm_fname = '/home/adama/SPT/hardware_maps_southpole/2019/hwm_pole/hwm.yaml'
    new_hwm_fname = '/home/adama/SPT/hardware_maps_southpole_test/2019/hwm_pole/hwm.yaml'

    old_hwm = pydfmux.load_session(open(old_hwm_fname, 'rb'))['hardware_map']
    bolos = old_hwm.query(pydfmux.Bolometer)
    old_mapping = {}
    for b in bolos:
        old_mapping[b.name] = {'physical_name': b.physical_name, 'pstring': b.pstring()}

    new_hwm = pydfmux.load_session(open(new_hwm_fname, 'rb'))['hardware_map']
    bolos = new_hwm.query(pydfmux.Bolometer)
    new_mapping = {}
    for b in bolos:
        new_mapping[b.name] = {'physical_name': b.physical_name, 'pstring': b.pstring()}

    with open('hwm_old.pkl', 'wb') as f:
        pickle.dump(old_mapping, f)
    with open('hwm_new.pkl', 'wb') as f:
        pickle.dump(new_mapping, f)
else:
    with open('hwm_old.pkl', 'rb') as f:
        old_mapping = pickle.load(f)
    with open('hwm_new.pkl', 'rb') as f:
        new_mapping = pickle.load(f)

# check whether the same names are present in both HWMs
if set(old_mapping.keys()) == set(new_mapping.keys()):
    print('Bolometer names are the same between new and old mappings.')
else:
    print('ERROR: Bolo names are not the same!')

# check whether the name <-> pstring mapping is the same in both
old_pstrings = []
new_pstrings = []
for bname in old_mapping:
    old_pstrings.append(old_mapping[bname]['pstring'])
    new_pstrings.append(new_mapping[bname]['pstring'])

if old_pstrings == new_pstrings:
    print('Bolometer pstrings are the same between new and old mappings.')
else:
    print('ERROR: Bolo pstrings are not the same!')


# check that the physical names that should have changed, did in fact change,
# and that other physical names did not
for bname in old_mapping:
    if old_mapping[bname]['physical_name'] != new_mapping[bname]['physical_name']:
        print('{} -> {}'.format(old_mapping[bname]['physical_name'], new_mapping[bname]['physical_name']))
