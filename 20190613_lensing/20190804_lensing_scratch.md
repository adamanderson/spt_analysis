---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 3 (v2)
    language: python
    name: python3-v2
---

# Scratchwork to Learn about Lensing Estimation
Zhaodi has been refactoring the lensing pipeline in the `lensing_restructure` branch. This hasn't yet been merged into master, but development has dropped off recently, so I'm guessing that it's in a relatively finalized state.

```python
# import packages.
# I have committed a copy of lensing code to spt3g_software.
# For now probably just add my build to the system path, which tested to work. 
# The lensing software doesn't interface with other parts of 3G software much.
import sys
# note: this is a frozen copy of software that lensing software works on
# people have work on serious code changes afterwards
#sys.path.append("/scratch/panz/lens100d_py3/scripts/spt3g_software/build/")
from builtins import zip
import os, sys, scipy, hashlib, glob, subprocess, imp, pdb
import pickle as pk
import healpy as hp
import numpy  as np
import pylab  as pl
import datetime
import glob
from spt3g import lensing as sl
```

```python
#Ell-space cuts
lmax_sinv    = 4000 # cutoff the filtered signal at this value                                          
lmax_cinv    = 3000 # cutoff the cinv-filtered fields at this value                                     
lx_cut       = 450  # mask modes below this value in lx  
#T to P leakage and polarization calibration coefficients
tq_leak= -0.0050
tu_leak= 0.0083
pcal = 1.048
# Name prefixes, used to name the data folders.
ivfs_prefix = "run08"
qest_prefix = "mf100_lx450_lmax3000"
#Specify which sims are used for which quadratic cl estimator
nsim = 500
mc_sims         = np.arange(0,nsim)# all sims' indices
mc_sims_mf      = mc_sims[0:100]   # sims used for mean field estimation
mc_sims_var     = mc_sims[100:500] # variance
mc_sims_unl     = mc_sims_var      # unlensed sims
mc_sims_qcr_mc  = mc_sims_var      # qcr, normalization calculation
mc_sims_n0      = mc_sims          # n0 bias
mc_sims_n1      = mc_sims[0:50]    # n1 bias
# The place for the input data and output data
bdir= '/scratch/panz/lens100d_py3/'
```

```python
# load up make_cmbs_scan.py, which has information about the sims and their locations.
scan        = imp.load_source("scan", bdir+"scripts/make_cmbs_scan.py")
# scan.cmbs contain the lensed, unlensed map power spectrum and the lensing power spectrum. 
cmbs        = scan.cmbs
# Define some map parameters stored in make_cmbs_scan.py
reso_rad    = np.double(scan.reso/60*np.pi/180.) # resolution: 2 arcmin                                             
npix        = int(scan.nx) # number of pixels along x or y: 390                                                                        
px          = sl.maps.pix(npix, reso_rad)   # pixel object in lensing software, contains map size and resolution.
lmax_theoryspec        = scan.lmax # 6000.  Cut off the theory spectrum beyond this ell 
```

```python

```
