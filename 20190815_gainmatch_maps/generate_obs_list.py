from adama_utils import get_obsids_from_times
from spt3g import core
import pandas as pd

obsids = get_obsids_from_times(core.G3Time("20190801_000000"),
                               core.G3Time("20190901_000000"),
                               sources=['ra0hdec-44.75', 'ra0hdec-52.25',
                                        'ra0hdec-59.75', 'ra0hdec-67.25'])

obsids_refactored = {'source':[], 'obsid':[]}
for source in obsids:
    for obsid in obsids[source]:
        obsids_refactored['source'].append(source)
        obsids_refactored['obsid'].append(obsid)

df = pd.DataFrame(obsids_refactored)
df.to_csv('obsids_to_process.txt', sep='\t', index=False)
