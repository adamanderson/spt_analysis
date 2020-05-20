import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.read_csv('_RT_data.csv', sep='\t')
tc = d['Tc_average']
band = np.array([int(pname.split('.')[1]) if 'resistor' not in pname else 0
                 for pname in d['physical_name']])
print(band)
plt.hist(tc[band==90], bins=np.linspace(0.41, 0.46, 51),
         histtype='step', label='90 GHz')
plt.hist(tc[band==150], bins=np.linspace(0.41, 0.46, 51),
         histtype='step', label='150 GHz')
plt.hist(tc[band==220], bins=np.linspace(0.41, 0.46, 51),
         histtype='step', label='220 GHz')
plt.hist(tc, bins=np.linspace(0.41, 0.46, 51),
         color='k', histtype='step', label='all')
plt.legend()
plt.xlabel('Tc [K]')
plt.ylabel('bolometers')
plt.title('average Tc between heating and cooling sweeps')
plt.xlim([0.41, 0.45])
plt.tight_layout()
plt.savefig('_Tc_average.png')
