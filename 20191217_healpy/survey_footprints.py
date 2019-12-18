from healpy import fitsfunc, visufunc
import numpy as np
import matplotlib.pyplot as plt
import argparse as ap
import os

parser = ap.ArgumentParser(description='Plots various survey footprints on the '
                           'Planck dust map.',
                           formatter_class=ap.ArgumentDefaultsHelpFormatter)
parser.add_argument('--download-data', action='store_true',
                    help='Download data for surveys that have nontrivial '
                    'footprint geometries.')
parser.add_argument('--show-coords', action='store_true',
                    help='Show coordinates to guide the eye.')
parser.add_argument('--surveys', action='store', type=str, nargs='*',
                    choices=['spt3g', 'sptpol', 'sptsz', 'bicep2', 'des'],
                    default=['spt3g', 'sptpol', 'sptsz', 'bicep2', 'des'],
                    help='Names of survey footprints to plot.')
parser.add_argument('--projection', action='store', type=str,
                    choices=['orthographic', 'mollweide'],
                    help='Projection to use')
args = parser.parse_args()


# Miscellaneous things
survey_names     = {'spt3g':'SPT-3G', 'sptpol':'SPTpol', 'sptsz':'SPT-SZ',
                    'bicep2':'BICEP2', 'des':'DES'}
survey_colors    = {'spt3g':'C1', 'sptpol':'C0', 'sptsz':'C2',
                    'bicep2':'C3', 'des':'C6'}


# Download data
if args.download_data:
    os.system('wget https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/foregrounds/COM_CompMap_dust-commander_0256_R2.00.fits')
    os.system('wget http://bicepkeck.org/B2_3yr_373sqdeg_field_20140509.txt')
    os.system('wget http://stash.ci-connect.net/spt/public/adama/survey_footprints/spider_obs_combined_outline.fits')
    os.system('wget http://stash.ci-connect.net/spt/public/adama/survey_footprints/des_footprint.txt')

# Load data
dust_map = fitsfunc.read_map('COM_CompMap_dust-commander_0256_R2.00.fits')
b2_footprint = np.loadtxt('B2_3yr_373sqdeg_field_20140509.txt', delimiter=',')
spider_footprint = fitsfunc.read_map('/spt/public/adama/survey_footprints/spider_obs_combined_outline.fits')
des_footprint = np.loadtxt('des_footprint.txt')


# Define the footprints
# SPT-3G survey
ra = {}
dec = {}
ra['spt3g']      = np.hstack([np.linspace(-50, 50, 1000), np.linspace(50, 50, 1000),
                              np.linspace(50, -50, 1000), np.linspace(-50, -50, 1000)])
dec['spt3g']     = np.hstack([np.linspace(-42, -42, 1000), np.linspace(-42, -70, 1000),
                              np.linspace(-70, -70, 1000), np.linspace(-70, -42, 1000)])

# SPTpol survey (see p3 https://arxiv.org/pdf/1707.09353.pdf)
ra['sptpol']     = np.hstack([np.linspace(-360*2/24, 360*2/24, 1000), np.linspace(360*2/24, 360*2/24, 1000),
                              np.linspace(360*2/24, -360*2/24, 1000), np.linspace(-360*2/24, -360*2/24, 1000)])
dec['sptpol']    = np.hstack([np.linspace(-50, -50, 1000), np.linspace(-50, -65, 1000),
                              np.linspace(-65, -65, 1000), np.linspace(-65, -50, 1000)])

# SPT-SZ survey (see p3 https://arxiv.org/pdf/1704.00884.pdf)
ra['sptsz']      = np.hstack([np.linspace(-360*4/24, 360*7/24, 1000), np.linspace(360*7/24, 360*7/24, 1000),
                              np.linspace(360*7/24, -360*4/24, 1000), np.linspace(-360*4/24, -360*4/24, 1000)])
dec['sptsz']     = np.hstack([np.linspace(-40, -40, 1000), np.linspace(-40, -65, 1000),
                              np.linspace(-65, -65, 1000), np.linspace(-65, -40, 1000)])

# BK "field outline" (from http://bicepkeck.org/B2_3yr_373sqdeg_field_20140509.txt)
ra['bicep2']     = b2_footprint[:,0]
dec['bicep2']    = b2_footprint[:,1]

# DES footprint (from spt.uchicago.edu: /home/cvsroot/spt_analysis/idlsave/round13-poly.txt)
# Need to interpolate more densely to deal with bug in orthographic plotting in healpy
ra['des']         = des_footprint[:,0]
dec['des']        = des_footprint[:,1]
ra['des']         = np.interp(np.linspace(0, len(ra['des']), 4*len(ra['des'])), np.arange(len(ra['des'])), ra['des'])
dec['des']        = np.interp(np.linspace(0, len(dec['des']), 4*len(dec['des'])), np.arange(len(dec['des'])), dec['des'])


# Do the plotting
plt.figure(figsize=(6,6))
if args.projection == 'orthographic':
    visufunc.orthview(dust_map, rot=(0, -55, 180),
                      norm='log', coord='GC', half_sky=True, notext=True, cbar=False, title='')
elif args.projection == 'mollweide':
    visufunc.mollview(dust_map,
                      norm='log', coord='GC', notext=True, cbar=False, title='')

visufunc.graticule(15, 360/12)
for survey in args.surveys:
    visufunc.projplot(ra[survey], dec[survey],
                      survey_colors[survey], lonlat=True, label=survey_names[survey])

if args.show_coords:
    visufunc.projtext(0, -45, '$-45^\circ$', lonlat=True, color='w', fontsize=14)
    visufunc.projtext(-4.2/24*360, -5, '$20^h$', lonlat=True, color='w', fontsize=14)
    visufunc.projtext(3.8/24*360, -5, '$4^h$', lonlat=True, color='w', fontsize=14)

plt.legend(fontsize=14, loc=(1.0,0.))
plt.savefig('survey_footprints.png', dpi=200)
