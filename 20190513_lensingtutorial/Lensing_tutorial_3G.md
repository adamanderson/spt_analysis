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

# Lensing tutorial, part 1, SPT3G Hackathon 


### This tutorial is based on the lens100d analysis. 


### Import packages

```python
# import packages.
# I have committed a copy of lensing code to spt3g_software.
# For now probably just add my build to the system path, which tested to work. 
# The lensing software doesn't interface with other parts of 3G software much.
import sys
sys.path.append("/home/panz/code/spt3g_software/build/")
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

### Define related parameters

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

### Load make_cmbs_scan.py.

We use simulations to estimate the variance, noise biases, and the normalization of the lensing power spectrum. The simulations are treated similarly to the data and have the same transfer function as the data. This requires using mock-observed simulations. 

The mock-observed SPTpol simulations used for this analysis were saved in numpy data arrays. The code make_cmbs_scan.py is just a code to load it up. It interfaces with cmb.py, which contain libraries to format the data into the corresponding simulation map objects. 


For SPT3G, we will also need to generate simulations and the mock-observed simulation maps. SPT3G has different format files (.g3 files), so this interface will change.

We chose the SPTpol 100d field for the tutorial because it's faster to run. 

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

### Plot the tmap from one sim.
* Note that the map is in unit of uK.
* Exercise: try plotting the q and u map as well. 

`tqu_1` in the code cell below is a `spt3g.lensing.lensing.maps.tqumap` object defined in `lensing/lensing/maps.py`.

# ```python
class tqumap(pix):
    def __init__(self, nx, dx, maps=None, ny=None, dy=None):
# ``` 

It has useful methods, such as `get_teb()`,
# ```python
    def get_teb(self):
        """ return a tebfft object containing the fourier transform of the T,Q,U maps. """
        ret = tebfft( self.nx, self.dx, ny = self.ny, dy = self.dy )

        lx, ly = ret.get_lxly()
        tpi  = 2.*np.arctan2(lx, -ly)
        #tpi= 2.*np.arctan2(ly, lx)
        tfac = np.sqrt((self.dx * self.dy) / (self.nx * self.ny))
        qfft = np.fft.rfft2(self.qmap) * tfac
        ufft = np.fft.rfft2(self.umap) * tfac

        ret.tfft[:] = np.fft.rfft2(self.tmap) * tfac
        ret.efft[:] = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
        ret.bfft[:] = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
        return ret
# ```
which will return a `spt3g.lensing.lensing.maps.tebfft` object.

# ```python
class tebfft(pix):
    def __init__(self, nx, dx, ffts=None, ny=None, dy=None):
# ```

`tebfft` also has a lot of useful methods, such as multiplication(\_\_mul\_\_), adding(\_\_add\_\_), get_tqu (convert to real space tqu maps), get_pix_transf(pixel window function), get_l_masked(mask modes above given ell).    

`tqumap` and `tebfft` are the TQU maps and their Fourier space TEB modes that the lensing pipeline interface with.

```python
tqu_1 = scan.cmblib_len_t1p1_scan.get_sim_tqu(1) # extract simulation indexed 1
pl.imshow(tqu_1.tmap[::-1,:], vmin= -400, vmax = 400) # plot in IAU convention
pl.title('T map of lensed sim 0 ($\mu K$)')
pl.colorbar()
tqu_1 # show the type
```

### Beam, transfer function, and foreground.

The cell below mostly load up things. I only want to mention the foreground library (`library_t`) we used, which is located in `lensing/lensing/sims/fgnd.py`.

We construct the power spectrum of the foregrounds, including sources, CIB, and TSZ, by:

# ```python
self.clsrc = sl.util.dictobj({'ls': ls, 'lmax': lmax, 'cltt': dl2cl * asrc * (ls / 3000.)**2})
self.clcib = sl.util.dictobj({'ls': ls, 'lmax': lmax, 'cltt': dl2cl * acib * (ls / 3000.)**pcib})
self.cltsz = sl.util.dictobj({'ls': ls, 'lmax': lmax, 'cltt': dl2cl * atsz * sztmpl[0:lmax + 1]})

# ```
and then convert the power spectrum to tebfft objects and add them on the sims.
# ```python
teb = sl.sims.tebfft(self.pix, self.clsrc)
teb += sl.sims.tebfft(self.pix, self.clcib)
teb += sl.sims.tebfft(self.pix, self.cltsz)
# ```
This may be too simple for us. We are likely going to use something different for 3G. The lens500d analysis used a more comprehensive foreground library. 

```python
print("---- get beams")
# load up the beam for SPTpol 100d lensing analysis
blpk150     = pk.load(open(bdir+"inputs/beam/composite_beam_2012_13_bb_runlist_v3_150.pkl",'rb'), encod\
ing= 'latin1')
map_cal_factor = blpk150['cal']  # composite calibration number, = 0.8926                               
bl150       = blpk150['B_ell_hybrid'] / map_cal_factor

# convert the beam into an object in 2d Fourier space. 
#Fill up the 2-D Fourier plane with the same value for all the annulus with same ell.
tf_beam_cal = (sl.maps.tebfft(px.nx, px.dx, 3*[np.ones([px.ny,px.nx//2+1])]) * bl150).get_l_masked(lmax\
=lmax_theoryspec)

# load up the 2D transfer function
print("---- get TF")
tf_file     = bdir+"inputs/tf2d/tf2d_sptpol_poly4_lowpass4000_nobeam_ngrid390_reso2am_proj5_1000iter.npy"

# convert the transfer function into an tebfft object in 2D Fourier space
tf150       = ( sl.maps.tebfft(px.nx, px.dx, 3*[np.load(tf_file) * sl.maps.tebfft(px.nx, px.dx).get_pix\
_transf().fft]) * bl150 ).get_l_masked(lmax=lmax_theoryspec)

# The forebround library, with CIB contribution.
print("---- get fgndlib")
fgndlib150  = sl.sims.fgnd.library_t(px, lmax_theoryspec, bdir+"qest/fgnds", asrc=10., acib=5., atsz=5.\
0, pcib=0.8,seed=44) ## 

print("---- get mask")
apod150      = np.load(bdir+"inputs/mask/apod_surv_five_150.npy") # used for plotting purposes only     
mask150      = np.load(bdir+"inputs/mask/mask_surv_five_150.npy") * np.load(bdir+"inputs/mask/mask_clus\
t10.npy") * np.load(bdir+"inputs/mask/mask_srccr10.npy") ##
```

Try plotting the beam and the transfer function. 

```python
pl.plot(blpk150['B_ell'], blpk150['B_ell_hybrid'])
pl.xlabel('Ell')
pl.ylabel('$B_l$')
pl.title('Beam in ell space')
pl.figure(2)
# only plot half of the 2D transfer function. The other half is similar.
pl.imshow(tf150.tfft[0:npix//2,:][::-1,:], extent = [0,5400, 0 ,5400])
pl.xlabel('lx')
pl.ylabel('ly')
pl.title('Transfer function')
```

### Observation libraries

Below are observation objects defined in `lensing/lensing/sims/obs.py`. 

`obs.library_half_noise_3g` pulls a data map or a sim map for you. 
# ```python
class library_half_noise_3g(object):
    '''
    Library for data and simulated observations, using half-observation differencing to estimate noise.
    Select Inputs:
        sky_lib,      The simulation library defined in make_cmbs_scan.py. It contains the sim location.
        half_dir,     The location for the data half coadds. Used to generate the noise realizations.
        tq,           T-Q leakage term
        tu,           T-U leakage term
        fp,           if fp=True, call flattenPol()
        thresh,       cut pixels in sim-maps with an absolute value greater than "thresh" [units of uK]
        tcal,         [float] the absolute calibration of T,Q,and U in map units.
        pcal,         [float] the calibration of Q and U maps.  This is applied to Q&U in addition to tcal.
    Note: In some analyses, the tcal factor is included in the beam in "transf".
    '''
    def __init__(self, pix, transf, sky_lib, half_dir, tq=0.0, tu=0.0, qu=0.0, fp=True, thresh=1000.,
                 tcal=1.0, pcal=1.0):
# ```
The `get_data_tqu` method will read in a data map for you. It will also apply the T->P leakage corrections and calibrations for the data.

The `get_sim_tqu` method will pull a sim and then add noise generated by differencing two half coadds to it. 

# ```python
    def get_sim_tqu(self, isim):
        '''
        Return the simulated sky + noise
        Note: noise (but not sim cmb skies) should be scaled by the calibration factors.
        '''
        halfA = readmap(self.half_dir + "/half_A_%d.g3" % (isim))[0]
        halfB = readmap(self.half_dir + "/half_B_%d.g3" % (isim))[0]
        hmap = self.fixmap(subtract(halfA, halfB))
        ... # Calibration...
        hmap_tqu = sl.maps.PolarizedMap_get_tqu(hmap)
        cmb_tqu = self.sky_lib.get_sim_tqu(isim)
        return (cmb_tqu.get_teb() * self.transf).get_tqu() + hmap_tqu
# ```


`obs.libarary_homogeneous_noise_3g` pulls a sim and add white noise to it. 
# ```python
    def get_sim_tqu(self, isim):
        '''
        Return the simulated sky + Gaussian realization of white noise
        '''
        cmb_tqu = self.sky_lib.get_sim_tqu(isim)

        # Apply a threshold to the maps
        cmb_tqu = cmb_tqu.threshold(self.thresh)

        # Add the noise realizations
        nx, dx, ny, dy = cmb_tqu.nx, cmb_tqu.dx, cmb_tqu.ny, cmb_tqu.dy

        import sptpol_software.lensing as sl
        hmap = sl.maps.tqumap(nx, dx, ny=ny, dy=dy)
        hmap.tmap = np.random.standard_normal(cmb_tqu.tmap.shape) * self.nlev_t / \
            (180. * 60. / np.pi * np.sqrt(dx * dy))
        hmap.qmap = np.random.standard_normal(cmb_tqu.qmap.shape) * self.nlev_p / \
            (180. * 60. / np.pi * np.sqrt(dx * dy))
        hmap.umap = np.random.standard_normal(cmb_tqu.umap.shape) * self.nlev_p / \
            (180. * 60. / np.pi * np.sqrt(dx * dy))

        return (cmb_tqu.get_teb() * self.transf).get_tqu() + hmap
# ```

```python
# halfs_dir is the directory for storing random half coadds of the maps,
# which will be used for generating the noise by differencing.
halfs_dir   = bdir+"data/map/20140418_ra23h30dec55_lens100d/halfs/150/"

# lensed sims or the data, scan.cmblib_len_t1p1_scan is the lensed sim library.
obs150_len  = sl.sims.obs.library_half_noise_3g( px, tf_beam_cal, sl.sims.library_tqu_sum( [scan.cmblib\
_len_t1p1_scan, fgndlib150] ), halfs_dir, tq=tq_leak, tu=tu_leak, thresh=1000., pcal=pca\
l )

# unlensed sims, scan.cmblib_unl_t1p1_scan is the unlensed sim library.
obs150_unl  = sl.sims.obs.library_half_noise_3g( px, tf_beam_cal, sl.sims.library_tqu_sum( [scan.cmblib\
_unl_t1p1_scan, fgndlib150] ), halfs_dir, tq=tq_leak, tu=tu_leak, thresh=1000., pcal=pca\
l )

# For n1 bias, nofg means no foreground. t2p1 means same lensing realizations but different cmb realizations.
# no foreground is needed for n1 bias
obs150_len_t2_nofg = sl.sims.obs.library_homogeneous_noise_3g( px, tf_beam_cal, sl.sims.library_tqu_sum\
( [scan.cmblib_len_t2p1_scan] ), thresh=1000.)

# For n1 bias, nofg means no foreground. t1p1 means different lensing realizations and different cmb realizations                                              
obs150_len_nofg    = sl.sims.obs.library_homogeneous_noise_3g( px, tf_beam_cal, sl.sims.library_tqu_sum\
( [scan.cmblib_len_t1p1_scan] ), thresh=1000.)
obslibs = [obs150_len, obs150_len_nofg, obs150_len_t2_nofg]
```

### Plot one lensed sim and one corresponding unlensed sim.  

Weak lensing is a surface brightness conserving remapping of
source to image planes by the gradient of the projected potential.

$x(\hat{n})$ (unlensed map) → $x(\hat{n} + ∇\phi)$ (lensed map)

Here $\phi$ is the lensing potential that we are trying to reconstruct here.


```python
pl.figure(figsize = (8,4))
pl.subplot(121)
tmap_lensed = obs150_len.get_sim_tqu(1).tmap
tmap_unlensed = obs150_unl.get_sim_tqu(1).tmap
#plot in IAU convention, the map unit is uK. Actually all the lensing software deals with map in uK. 
pl.imshow(tmap_lensed[::-1,:], vmin = -400, vmax = 400)
pl.title('Lensed T map')
pl.subplot(122)
#plot in IAU convention, the map unit is uK.
pl.imshow(tmap_unlensed[::-1,:], vmin = -400, vmax = 400)
pl.title('Unlensed T map')
pl.figure()
pl.imshow((tmap_lensed- tmap_unlensed)[::-1,:], vmin=-400, vmax = 400)
pl.title('Difference')
```

### Inverse-variance filtering

We need to inverse-variance filter the data maps. This step extract weighted Fourier modes to reduce the estimator scatter. When explaining how quadratic estimator works, we will say more about why the inverse variance filter can reduce the variance of the quadratic estimator.  

To explain this step, we firstly define the data maps and sky signal:
 * $d\in[T(\hat{\mathbf{n}}), Q(\hat{\mathbf{n}}), U(\hat{\mathbf{n}})]$ is the data matrix. $d_j$ is the map value of the $j$th pixel.
 * $X_{\mathbf{l}}\in[T(\mathbf{l}), E(\mathbf{l}), B(\mathbf{l})]$ is the TEB sky signal. 

The data maps are related to the sky signal by:

$d_j = \sum_{\mathbf{l}} P_{j\mathbf{l}}X_{\mathbf{l}}+\sum_{\mathbf{l}} P_{j\mathbf{l}}N_{\mathbf{l}}+ n_j$

  * Here $P_{j\mathbf{l}}$ is a matrix operater that includes the transfer function, Fourier Transform, and QU->EB conversion. 

    * For temperature the operator is:

      * $P_{j\mathbf{l}} = e^{i\mathbf{l}\mathbf{x}_j}F_{\mathbf{l}} $, where $F_{\mathbf{l}}$ is the transfer function, which includes beam, pixelization and timestream filtering.

    * For EB, this operator is:
 
      * $P_{j\mathbf{l}} = e^{i\mathbf{l}\mathbf{x}_j\pm 2i\phi_{\mathbf{l}}}F_{\mathbf{l}} $, which also includes the Fourier space angle $\phi_{\mathbf{l}}$ for QU-> EB conversion.

  * $N_{\mathbf{l}}$ is the sky noise that doesn't trace the CMB, such as foreground and atmosphere noise.

  * $n_j$ is the map noise. 

The formula for inverse-variance filtering is:

 $\overline X = S^{-1}[S^{-1}+P^{\dagger}n^{-1}P]^{-1} P^{\dagger}n^{-1}d$ 


 * $\overline X$ is the inverse-variance filtered fields. $\overline X = S^{-1} \hat{X}$, where $\hat{X}$ is the extimation of field $X$. 

 * $S = C_l^X + C_{\mathbf{l}}^N$, where $C_l^X$ is the theoretical power spectrum and $C_{\mathbf{l}}$ is the 2D power spectrum of the noise. 

 * $P$ is the "pointing matrix" which transfers the TQU data in real space into TEB modes in ell space. $P$ was defined above.

### Code implementation
The inverse noise matrix $n^{-1}$ is calculated by `lensing/lensing/maps.py`'s `make_tqumap_wt`. If the white noise level arguments `nlev_tp` is given, it will just return inverse white noise square.
# ```python
    if nlev_tp is not None:
        ret.weight = np.zeros( ret.weight.shape )
        ret.weight[:,:,0,0] = (180.*60./np.pi)**2 * ret.dx * ret.dy / nlev_tp[0]**2
        ret.weight[:,:,1,1] = (180.*60./np.pi)**2 * ret.dx * ret.dy / nlev_tp[1]**2
        ret.weight[:,:,2,2] = (180.*60./np.pi)**2 * ret.dx * ret.dy / nlev_tp[1]**2
# ```

$S^{-1}$ is calculated by `lensing/lensing/cinv/opfilt_teb.py`. 
# ```python
def cl2sinv(cl, clnk, tf2d, nft=0.0, nfp=0.0, ebeq=True, lmax=None):
    '''
    Arguments:
      cl,          signal spectrum
      clnk,        noise spectrum
      tf2d,        2d transfer function
      nft,         noise-floor for Temperature  [uK-arcmin]
      nfp,         noise-floor for Polarization [uK-arcmin]
    Returns: Best estimate of sky signal, including:
      a) signal, b) "sky noise" diagonal in k, and c) subtracted pixel noise (white in k)
    '''
    ...  

    ret = tf2d * tf2d * ( tf2d * tf2d * cl + clnk ).inverse()
    return ret.get_l_masked(lmax=lmax)
# ```


The cinv library lives in `lensing/lensing/sims/cinv.py`
# ```python
class library(object):

    '''
    Library for C-inv filtered maps.
       obs_lib,               the observation library.
                                  E.g., for sims, sims.cmb.py, library_scan
                                  E.g., for data, ??
    '''
# ```

The part that does the inverse-variance filtering is the `cache_teb` method.

# ```python
    def cache_teb(self, tfname, tqu_obs):
        '''
        Outputs teb_filt_cinv = C-inv filtered TEB: tbar, bbar, ebar in e.g. EBPhi notes
        cd_solve: Conjugate gradient method
                  x      = [fwd_op]^-1 b, where fwd_op may not be invertible
                  fwd_op = C^-1 + P^+ N^-1 P    (cf. eqn 31, EBPhi notes)
                  b      = P^+ N^-1 d
        '''
        assert(not os.path.exists(tfname))
        # sinv_filt and ninv_filter are the inputs.
        sinv_filt, ninv_filt = self.sinv_filt, self.ninv_filt
        pre_op = sl.cinv.opfilt_teb.pre_op_diag(sinv_filt, ninv_filt)
        monitor = sl.cinv.cd_monitors.monitor_basic(sl.cinv.opfilt_teb.dot_op(), iter_max=np.inf, eps_min=self.eps_min)

        teb_filt_cinv = sl.maps.tebfft(tqu_obs.nx, tqu_obs.dx, ny=tqu_obs.ny, dy=tqu_obs.dy)
        # this code lives in lensing/lensing/cinv/cd_solve.py
        # it solves the matrix inversion problem by doing it in iterations
        # the criterion for convergence of iterations is set by eps_min
        sl.cinv.cd_solve.cd_solve(x=teb_filt_cinv,
                                  b=sl.cinv.opfilt_teb.calc_prep(tqu_obs, sinv_filt, ninv_filt),
                                  fwd_op=sl.cinv.opfilt_teb.fwd_op(sinv_filt, ninv_filt),
                                  pre_ops=[pre_op], dot_op=sl.cinv.opfilt_teb.dot_op(),
                                  criterion=monitor, tr=sl.cinv.cd_solve.tr_cg, cache=sl.cinv.cd_solve.cache_mem())
        # the following returns C^-1 [C^-1 + P^+ N^-1 P]^-1 P^+ N^-1 d
        teb_filt_cinv = sl.cinv.opfilt_teb.calc_fini(teb_filt_cinv, sinv_filt, ninv_filt)

        assert(not os.path.exists(tfname))
        pk.dump(teb_filt_cinv, open(tfname, 'w'),protocol = pk.HIGHEST_PROTOCOL)
# ```

```python
print("---- ivfs:ninv150")
# define n^{-1}
ninv150      = sl.maps.make_tqumap_wt( px, ninv=obs150_len.get_ninv(), ninv_dcut=1.e-5, nlev_tp=(7.,7.), mask=mask150 )
ninvfilt150  = sl.cinv.opfilt_teb.ninv_filt( tf150, ninv150 ) # map-space noise                                                                                          

print("---- ivfs:sinvfilt150")
# need to delete clte, otherwise the iteration for solving matrix inversion won't converge
cl_len_filt = scan.cl_len.copy(lmax=lmax_sinv); del cl_len_filt.clte # theoretical spectrum                                                                              
# includes signal and fourier-space noise 
# calculate S^{-1}
sinvfilt150 = sl.cinv.opfilt_teb.cl2sinv( cl_len_filt + fgndlib150.get_clfg(lmax=lmax_sinv),
                                          pk.load(open(bdir+"inputs/nl2d/clnk_150.pk",'rb')), tf150, nft=7., nfp=7., lmax=lmax_sinv )

print("---- ivfs:cinv150")
# Here we do the inversion in iterations, eps_min=4.e-4 is the convergence criterion.
# inverse-variance filtered lensed sims (and data)
cinv150_len = sl.sims.cinv.library( obs150_len, sinvfilt150, ninvfilt150, bdir+"qest/%s/cinv150_len_t1p1/"%ivfs_prefix, eps_min=4.e-4 )
# mask out the high ell
cinv150_len = sl.sims.cinv.library_l_mask( lxmin=lx_cut, lmax=lmax_cinv, cinv=cinv150_len )
# inverse-variance filtered unlensed sims
cinv150_unl = sl.sims.cinv.library( obs150_unl, sinvfilt150, ninvfilt150, bdir+"qest/%s/cinv150_unl_t1p1/"%ivfs_prefix, eps_min=4.e-4 )
# mask out the high ell
cinv150_unl = sl.sims.cinv.library_l_mask( lxmin=lx_cut, lmax=lmax_cinv, cinv=cinv150_unl )

# lensed sims with no foreground
cinv150_len_t2_nofg = sl.sims.cinv.library( obs150_len_t2_nofg, sinvfilt150, ninvfilt150, bdir+"qest/%s/cinv150_len_t2p1_nofg/"%ivfs_prefix, eps_min=4.e-4 )
cinv150_len_t2_nofg = sl.sims.cinv.library_l_mask( lxmin=lx_cut, lmax=lmax_cinv, cinv=cinv150_len_t2_nofg )

# lensed sims with no foreground
cinv150_len_nofg = sl.sims.cinv.library( obs150_len_nofg, sinvfilt150, ninvfilt150, bdir+"qest/%s/cinv150_len_t1p1_nofg/"%ivfs_prefix, eps_min=4.e-4 )
cinv150_len_nofg = sl.sims.cinv.library_l_mask( lxmin=lx_cut, lmax=lmax_cinv, cinv=cinv150_len_nofg )


# Separate these out, since different numbers of sims are used for different purposes                                                                                    
ivflibs = [cinv150_len, cinv150_len_nofg, cinv150_len_t2_nofg] # everything, used for get_dat_teb()                                                                      
ivflibs_mc_sims     = [cinv150_len] # evaluate for idxs in mc_sims                                                                                                       
ivflibs_mc_sims_n1  = [cinv150_len_nofg, cinv150_len_t2_nofg] # evaluate for idxs in mc_sims_mf                                                                          
ivflibs_mc_sims_unl = [cinv150_unl] # evaluate for idxs in mc_sims_unl                                                                                                   

```

Do cinv for one sim and look at it.
Compare with the original data map. 

```python
cinv_1 = cinv150_len.get_sim_teb(1) # do inverse-variance filtering for sim with index 1
# smooth by a 10 arcmin beam because dividing by C_l+C_l^N blows up the small scale noise
# without smoothing only noise is visible
# also note that cinv only keeps modes with ell< 3000
bl = sl.lensing.spec.bl(10., 5000) 
cinv_2 = cinv_1* bl

pl.figure(figsize = (8,4))
pl.subplot(121)
tmap_lensed = obs150_len.get_sim_tqu(1).tmap
pl.imshow(cinv_2.get_tqu().tmap[::-1,:])
pl.colorbar()
pl.title('Cinv filtered T map, arb unit')
pl.subplot(122)
pl.imshow((apod150*tmap_lensed)[::-1,:], vmin = -400, vmax = 400)
pl.colorbar()
pl.tight_layout()
pl.title('T map before cinv filtering')
```

### Quadratic estimator

Below is a brief review of the quadratic estimator. We'll go through how it works and make connections to our implementations in the codes. 

We all know that weak lensing is a remapping of source to image planes by the gradient of the projected potential. Below we take the temperature field as an example. 

$T(\hat{\mathbf{n}})$ = $\tilde{T}(\hat{\mathbf{n}} + ∇\phi(\hat{\mathbf{n}})) = \tilde{T}(\hat{\mathbf{n}})+ \nabla _i \phi(\hat{\mathbf{n}})\nabla^{i}\tilde{T}(\hat{\mathbf{n}})+ $ second order terms

Here $T$ is the lensed field, $\tilde{T}$ is the unlensed field, and $\phi$ is the projected lensing potential. 

$T(\hat{\mathbf{n}})$ and $\phi(\hat{\mathbf{n}})$ can be related to their Fourier Transforms by: 

* $T(\hat{\mathbf{n}}) = \int\frac{d^2 l }{{2\pi}^2}T(\mathbf{l})e^{i\mathbf{l}\hat{\mathbf{n}}}$,  $T({\mathbf{l}}) = \int d \hat{\mathbf{n}} T(\hat{\mathbf{n}})e^{-i\mathbf{l}\hat{\mathbf{n}}}$
* $\phi(\hat{\mathbf{n}}) = \int\frac{d^2 l }{{2\pi}^2}\phi(\mathbf{l})e^{i\mathbf{l}\hat{\mathbf{n}}}$,  $\phi({\mathbf{l}}) = \int d \hat{\mathbf{n}} \phi(\hat{\mathbf{n}})e^{-i\mathbf{l}\hat{\mathbf{n}}}$.

We expand the lensed field $T(\mathbf{l})$:

* $T(\mathbf{l}) = \int d \hat{\mathbf{n}} T(\hat{\mathbf{n}})e^{-i\mathbf{l}\hat{\mathbf{n}}} = \int d \hat{\mathbf{n}} [\tilde{T}(\hat{\mathbf{n}})+ \nabla _i \phi(\hat{\mathbf{n}})\nabla^{i}\tilde{T}(\hat{\mathbf{n}})]  e^{-i\mathbf{l}\hat{\mathbf{n}}} = \tilde{T}(\mathbf{l}) - \int \frac{d\mathbf{l_1}}{{2\pi}^2}\mathbf{l_1}\tilde{T}(\mathbf{l_1})(\mathbf{l-l_1})\phi(\mathbf{l-l_1})$

Here we have used the convolution theorem when calculating the Fourier Transform of $\nabla _i \phi(\hat{\mathbf{n}})\nabla^{i}\tilde{T}(\hat{\mathbf{n}})$. 

The power spectrum of temperature is defined as:
* $<\tilde{T}^*(\mathbf{l})\tilde{T}(\mathbf{l^{\prime}})> = (2\pi)^2 \delta(\mathbf{l}-\mathbf{l^{\prime}})\tilde C_{l}^{TT}$


Using the previous two equations, we have:

* $<T(\mathbf{l})T(\mathbf{l^{\prime}})>_{CMB} = [\tilde{C}_l^{TT}(\mathbf{l\cdot L})+\tilde{C}_{l^{\prime}}^{TT}(\mathbf{l^{\prime}\cdot L})]\phi(\mathbf{L}) = f(\mathbf{l},\mathbf{l^{\prime}}) \phi(\mathbf{L})$ 
  * $ f(\mathbf{l},\mathbf{l^{\prime}})= \tilde{C}_l^{TT}(\mathbf{l\cdot L})+\tilde{C}_{l^{\prime}}^{TT}(\mathbf{l^{\prime}\cdot L})$
  *  Here $\mathbf{L=l+l^{\prime}}$ and $\mathbf{L}\neq \mathbf 0 $.
  *  $<>_{CMB}$ means we only average over the CMB realizations and not the LSS, otherwise $\phi(\mathbf{L})$ would average to 0. 

The power spectrum of $\phi$ is:

* $<{\phi^*}(\mathbf{l})\phi(\mathbf{l^{\prime}})> = (2\pi)^2 \delta(\mathbf{l}-\mathbf{l^{\prime}}) C_l^{\phi\phi}$
    * Note that <> means averaging over CMB and LSS(lensing) realizations. 

We define a weighting of the moments to extract the lensing potential

*  $\hat{\phi}(\mathbf{L}) = R(\mathbf{L})^{-1} \int \frac{d^2\mathbf{l}}{{2\pi}^2}F(\mathbf{l}, \mathbf{l^{\prime}}) T(\mathbf{l})T(\mathbf{l^{\prime}})$
  * Again $\mathbf{L=l+l^{\prime}}$ and $\mathbf{L}\neq \mathbf 0 $.
  * Here $R(\mathbf{L})^{-1}$ is the normalization factor chosen such that $<\hat{\phi}(\mathbf{L})>_{CMB} = \phi (\mathbf{L})$

  * The form of $F(\mathbf{l}, \mathbf{l^{\prime}})$ that minimizes the variance $<\hat{\phi}^{*}(\mathbf{L})\hat{\phi}(\mathbf{L})>- <\hat{\phi}^{*}(\mathbf{L})><\hat{\phi}(\mathbf{L})>$ is:
    * $F(\mathbf{l}, \mathbf{l^{\prime}}) = f(\mathbf{l},\mathbf{l^{\prime}})/(2C_{l}^{TT}C_{l^{\prime}}^{TT}) $ 
  * Exercise: calculate the variance and try to minimize it. Verify the above $F$ minimizes the variance. 
* Several notes:
  * The form of $F(\mathbf{l}, \mathbf{l^{\prime}})$ is obtained by minimizing the variance of the estimated $\hat{\phi}$. 
  * $F(\mathbf{l}, \mathbf{l^{\prime}})$ is proportional to $f(\mathbf{l}, \mathbf{l^{\prime}})$ divided by the power spectrum of the fields used for the estimation. If the field contains sky noise $C_{\mathbf{l}}^N$, then the $C_{l}^{TT}$ in the denominator needs to be changed to $S = C_{l}^{TT}+C_{\mathbf{l}}^N$. 
  * **In our implementation of the quadratic estimator we define the weighting function to be $W(\mathbf{l}, \mathbf{l^{\prime}})= f(\mathbf{l}, \mathbf{l^{\prime}})/2$. The inverse-variance filtering $S^{-1}$ takes care of the denominator in $F(\mathbf{l}, \mathbf{l^{\prime}})$.**
  * TE, EE, EB, and TB estimators are derived similarly, but a little more complicated due to the ell space angle in QU->EB conversion.
  * Exercise: from $Q\pm iU= -(E\pm iB)e^{\pm 2 i \phi_{\mathbf{l}}}$ and $[Q\pm iU](\hat{\mathbf{n}})= [\tilde{Q}\pm i\tilde{U}](\hat{\mathbf{n}})+ \nabla _i \phi(\hat{\mathbf{n}})\nabla^{i}[\tilde{Q}\pm i\tilde{U}](\hat{\mathbf{n}})$, derive $f(\mathbf{l}, \mathbf{l^{\prime}})$ for $<T(\mathbf{l})E(\mathbf{l^{\prime}})>_{CMB} = f(\mathbf{l}, \mathbf{l^{\prime}}) \phi(\mathbf{l+l^{\prime}})$
  
### Biases in the power spectrum estimated from the  quadratic estimator

We estimate the power spectrum by $<\hat{\phi}^{*}(\mathbf{L})\hat{\phi}(\mathbf{L})>$

 * Here $\hat{\phi}(\mathbf{L}) = A(L) \int \frac{d^2\mathbf{l}}{{2\pi}^2}F(\mathbf{l}, \mathbf{l^{\prime}}) T(\mathbf{l})T(\mathbf{l^{\prime}})$. 

 * $\hat{\phi}$ means the estimated $\phi$. 

Note that conjugate in Fourier space is equivalent to switching $\mathbf{l}$ to $\mathbf{-l}$.

$<\hat{\phi}^{*}(\mathbf{L})\hat{\phi}(\mathbf{L^{\prime}})>$ is proportional to $\int d^2\mathbf{l_1}\int d^2 \mathbf{l_1^{\prime}}\, F(\mathbf{l_1}, \mathbf{l_2})F(\mathbf{l_1^{\prime}}, \mathbf{l_2^{\prime}})<T(\mathbf{-l_1})T(\mathbf{-l_2})T(\mathbf{l_1^{\prime}})T(\mathbf{l_2^{\prime}})>$

* Here $<T(\mathbf{-l_1})T(\mathbf{-l_2})T(\mathbf{l_1^{\prime}})T(\mathbf{l_2^{\prime}})>$ is a four-point correlation function. 

* $\mathbf{L}=\mathbf{l_1}+\mathbf{l_2}$

* $\mathbf{L^{\prime}}=\mathbf{l_1^{\prime}}+\mathbf{l_2^{\prime}}$

Using Wick's theorem, 

$<T(\mathbf{-l_1})T(\mathbf{-l_2})T(\mathbf{l_1^{\prime}})T(\mathbf{l_2^{\prime}})>$  =  $<<T(\mathbf{-l_1})T(\mathbf{-l_2})T(\mathbf{l_1^{\prime}})T(\mathbf{l_2^{\prime}})>_{CMB}>_{LSS}$ = 

$<<T(\mathbf{-l_1})T(\mathbf{-l_2})>_{CMB}<T(\mathbf{l_1^{\prime}})T(\mathbf{l_2^{\prime}})>_{CMB}>_{LSS}$    -----(Term 1)

$+ <<T(\mathbf{-l_1})T(\mathbf{l_1^{\prime}})>_{CMB}<T(\mathbf{-l_2})T(\mathbf{l_2^{\prime}})>_{CMB}>_{LSS}$    -----(Term 2)

$+ <<T(\mathbf{-l_1})T(\mathbf{l_2^{\prime}})>_{CMB}<T(\mathbf{-l_2})T(\mathbf{l_1^{\prime}})>_{CMB}>_{LSS}$    -----(Term 3)

Remember that
 * $<T(\mathbf{l})T(\mathbf{l^{\prime}})>_{CMB} = C_{|\mathbf {l+l^{\prime}}|}\delta(\mathbf{l+l^{\prime}})+f(\mathbf{l},\mathbf{l^{\prime}}) \phi(\mathbf{L})$+ higher order terms
 * $<{\phi^*}(\mathbf{l})\phi(\mathbf{l^{\prime}})>_{LSS} = (2\pi)^2 \delta(\mathbf{l}-\mathbf{l^{\prime}}) C_{l}^{\phi\phi}$

Now we figure out each term:

* Term 1 = $(2\pi)^4 C_{l_1}^{TT}C_{l_1^{\prime}}^{TT} \delta(\mathbf{L}) \delta(\mathbf{L^{\prime}}) + C_{L}^{\phi\phi} f(\mathbf{l_1},\mathbf{l_2})f(\mathbf{l_1^{\prime}},\mathbf{l_2}^{\prime})$

* Term 2 = $(2\pi)^4 C_{l_1}^{TT}C_{l_2}^{TT} \delta(\mathbf{l_1^{\prime}-l_1}) \delta(\mathbf{l_2^{\prime}-l_2}) + C_{|\mathbf {l_1-l_1^{\prime}}|}^{\phi\phi} f(\mathbf{-l_1},\mathbf{l_1^{\prime}})f(\mathbf{-l_2},\mathbf{l_2}^{\prime})$

* Term 3 = $(2\pi)^4 C_{l_1}^{TT}C_{l_2}^{TT} \delta(\mathbf{l_2^{\prime}-l_1}) \delta(\mathbf{l_1^{\prime}-l_2}) + C_{|\mathbf {l_1-l_2^{\prime}}|}^{\phi\phi} f(\mathbf{-l_1},\mathbf{l_2^{\prime}})f(\mathbf{-l_2},\mathbf{l_1}^{\prime})$

Because $\mathbf{L}\neq \mathbf{0}$, the first term of Term 1 vanishes. The second term in Term 1 gives the $C_L^{\phi\phi}$ reconstruction and is the signal we want to extract.

Term 2 and Term 3 generate $n_0$ bias and $n_1$ bias:

 * The first term in Term 2 and the first term in Term 3 are 0th order of $C_{L}^{\phi\phi}$, and they give rise to the $n_0$ bias. 

 * The second term in Term 2 and the second term in Term 3 are 1st order of $C_{|\mathbf {l_1-l_1^{\prime}}|}^{\phi\phi}$ (or $C_{|\mathbf {l_1-l_2^{\prime}}|}^{\phi\phi}$), and they give rise to the $n_1$ bias.


### Implementation of bias calculation in the code
In the code, we do not calculate the biases analytically. We calculate them from many MC realizations of simulations.

* $N_0$ bias:
   * $\Delta C_{\mathbf{L}}^{\phi\phi}|_{N_0} = <C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC},\bar{V}_{MC^{\prime}},\bar{X}_{MC},\bar{Y}_{MC^{\prime}}]+C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC},\bar{V}_{MC^{\prime}},\bar{X}_{MC^{\prime}},\bar{Y}_{MC}]>$

     * Here $MC$ and $MC^{\prime}$ are MC realizations with random CMB and random LSS (lensing field). 
     * <> means averaging over different $MC$ and $MC^{\prime}$ realizations. 
     * $U,V,X,Y \in \{TEB\}$
     * $<C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC},\bar{V}_{MC^{\prime}},\bar{X}_{MC},\bar{Y}_{MC^{\prime}}]>$ is calculated by $<\hat{\phi}^{UV*}(\mathbf{L})\hat{\phi}^{XY}(\mathbf{L^{\prime}})>$, which is proportional to $\int d^2\mathbf{l_1}\int d^2 \mathbf{l_1^{\prime}}\, F(\mathbf{l_1}, \mathbf{l_2})F(\mathbf{l_1^{\prime}}, \mathbf{l_2^{\prime}})<\bar U_{MC}(\mathbf{-l_1})\bar V_{MC^{\prime}}(\mathbf{-l_2})\bar X_{MC}(\mathbf{l_1^{\prime}})\bar Y_{MC^{\prime}}(\mathbf{l_2^{\prime}})>$
     * Using Wick's theorem we can expand the above 4-pt correlation function into three terms, just like in the previous section. 
     * The only non-vanishing part comes from $<\bar U_{MC}(\mathbf{-l_1}) \bar X_{MC}(\mathbf{l_1^{\prime}})><\bar V_{MC^{\prime}}(\mathbf{-l_2}) \bar Y_{MC^{\prime}}(\mathbf{l_2^{\prime}})>$, which corresponds to the first term in Term 2 (see previous section). Other terms average to zero because the correlation between $MC$ and $MC^{\prime}$ averages to zero. **Kimmy and I had some discussion whether the statement here is true. Please let us know if you have opinions.**
     * $MC$ and $MC^{\prime}$ have different $\phi$(LSS) realization, so there IS NO contribution to the first order of $C_L^{\phi\phi}$ ($\hat{\phi}^{MC, MC^{\prime}}$ average to zero). 
     * Similarly, $C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC},\bar{V}_{MC^{\prime}},\bar{X}_{MC^{\prime}},\bar{Y}_{MC}]$ corresponds to the first term in Term 3 (see previous section). 
     
* $N_1$ bias:
   * $\Delta C_{\mathbf{L}}^{\phi\phi}|_{N_1} = <C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC,\phi},\bar{V}_{MC^{\prime},\phi},\bar{X}_{MC,\phi},\bar{Y}_{MC^{\prime},\phi}]+C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC,\phi},\bar{V}_{MC^{\prime},\phi},\bar{X}_{MC^{\prime},\phi},\bar{Y}_{MC,\phi}] - \Delta C_{\mathbf{L}}^{\phi\phi}|_{N_0}>$

     * Here $(MC,\phi)$ and $(MC^{\prime},\phi)$ are MC realizations with random CMB but the same LSS (lensing field). 
     * $U,V,X,Y \in \{TEB\}$
     * Similar to the $N_0$ case, in $<C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC,\phi},\bar{V}_{MC^{\prime},\phi},\bar{X}_{MC,\phi},\bar{Y}_{MC^{\prime},\phi}]>$, the only non-vanishing part comes from $<\bar U_{MC}(\mathbf{-l_1}) \bar X_{MC}(\mathbf{l_1^{\prime}})><\bar V_{MC^{\prime}}(\mathbf{-l_2}) \bar Y_{MC^{\prime}}(\mathbf{l_2^{\prime}})>$, which corresponds to Term 2 (see previous section). 
     * $MC,\phi$ and $MC^{\prime},\phi$ have the same $\phi$(LSS) realization, so there IS contribution to the first order of $C_L^{\phi\phi}$. 
     * Similarly, $C_{\mathbf{L}}^{\hat{\phi}\hat{\phi}}[\bar{U}_{MC,\phi},\bar{V}_{MC^{\prime},\phi},\bar{X}_{MC^{\prime},\phi},\bar{Y}_{MC,\phi}]$ corresponds to Term 3 (see previous section).
     
### Implementation of quadratic estimator in the code

In the derivation above, we have

*  $\hat{\phi}(\mathbf{L}) = R(\mathbf{L})^{-1}\cdot \int \frac{d^2\mathbf{l}}{{2\pi}^2}F(\mathbf{l}, \mathbf{l^{\prime}}) T(\mathbf{l})T(\mathbf{l^{\prime}})$
    * Here $\mathbf{l^{\prime} } = \mathbf{L} - \mathbf{l} $

* $F(\mathbf{l}, \mathbf{l^{\prime}}) = f(\mathbf{l},\mathbf{l^{\prime}})/(2C_{l}^{TT}C_{l^{\prime}}^{TT}) $

In the lensing software we implement this in the following steps:

* Firstly we inverse-variance filter the fields: $\bar T= S^{-1} T$, which is essentially $\bar T = T/C_{l}^{TT}$ when there is no sky noise.

* We define our weighting function to be $W(\mathbf{l},\mathbf{l^{\prime}})=f(\mathbf{l},\mathbf{l^{\prime}})$. 

* We calculate our lensing reconstruction as
$\hat{\phi}(\mathbf{L}) =  R(\mathbf{L})^{-1} 2 \int \frac{d^2\mathbf{l}}{{2\pi}^2}W(\mathbf{l}, \mathbf{l^{\prime}}) \bar T(\mathbf{l})\bar T(\mathbf{l^{\prime}})$
 
   * Here $R_{\mathbf{L}}$ is the normalization function such that $<\hat{\phi}(\mathbf{L})>= \phi(\mathbf{L})$
   * The extra factor of 2 comes from the 2 in the denominator in $F$, which is NOT included in our $W$ function. 
   * The analytic form of $R^{Analytic}_{\mathbf{L}}$ is $R^{Analytic}_{\mathbf{L}}= \int d^2\mathbf{l}\,W(\mathbf{l}, \mathbf{l^{\prime}})W(\mathbf{l}, \mathbf{l^{\prime}}) F_{\mathbf l }F_{\mathbf {l^{\prime}} }$.
       * Here $F_{\mathbf l }$ is the transfer function, including filtering, pixelization, and beam. 
   * The analytic $R$ can be off because the transfer function may not be exact. We also use MC simulations to estimate $R$ from simulations. $R^{MC}(\mathbf{L}) = \frac{<\hat{\phi}(\mathbf{L})\phi^{I*}(\mathbf{L})>}{<\phi^{I}(\mathbf{L})\phi^{I*}(\mathbf{L})>}$
       * Here $\phi^{I*}$ is the input $\phi$ for the sims and $\hat{\phi}$ is the reconstructed $\phi$.
   * The full normalization is $R = R^{Analytic}R^{MC}$.
* Correct for the mean-field bias.
    * There may be some contributioon to our $\phi$ estimation from statistical anisotropy induced by non-lensing sources such as the mask and inhomogeneous noise. These terms consistute a "mean-field" bias.
    * We calculate the mean-field bias $\bar{\phi}_{\mathbf L}^{MF}$ by averaging over 50 simulations with independent realizations of CMB and LSS but the same mask and inhomogeneous noise. The lensing signal should average out, leaving us with the mean-field bias. In $\bar{\phi}$ the bar means it's not corrected for the normalization $R$.  
    * We can compare the mean field bias with $n_0$ and $n_1$ bias. The mean field bias is for $\phi$, while $n_0$ and $n_1$ are for $C_L^{\phi\phi}$.
* Putting in the mean-field bias and the MC correction of the normalization function, the unbiased estimation of the $\phi$ is $\hat{\phi}(\mathbf{L}) = R(\mathbf{L})^{-1} 2 [\int \frac{d^2\mathbf{l}}{{2\pi}^2}W(\mathbf{l}, \mathbf{l^{\prime}}) \bar T(\mathbf{l})\bar T(\mathbf{l^{\prime}})-\bar{\phi}_{\mathbf L}^{MF}]$


### Definition of the weight functions in the code
In the lensing pipeline, the weighting function $W$ for $TT, TE, EE, EB, TB$ estimators are defined in `spt3g_software/lensing/lensing/qest.py`

In our implementation, $W$ for TT estimator is:

$W_{\mathbf{l}, \mathbf{l^{\prime}}}= [\tilde{C}_l^{TT}(\mathbf{l\cdot L})+\tilde{C}_{l^{\prime}}^{TT}(\mathbf{l^{\prime}\cdot L})]\phi(\mathbf{L})$

$\mathbf{l}\cdot\mathbf{L} = Ll\cos(\theta_0-\theta_L) = Ll\frac 1 2 (e^{i\theta_0}e^{-i\theta_L}+e^{-i\theta_0}e^{i\theta_L})$
 * Here $\theta_0$ and $\theta_L$ are azimuthal angles of $l$ and $L$, respectively.

Expand $W$ as a function of azimuth angles of $\mathbf{l}, \mathbf{l^{\prime}}, \mathbf{L}$

$W_{\mathbf{l}, \mathbf{l^{\prime}}} = {[(\frac 1 2 L e^{-i\theta_L})\tilde{C}_l^{TT}le^{i\theta_0}]+[(\frac 1 2 L e^{i\theta_L})\tilde{C}_l^{TT}le^{-i\theta_0}]+[(\frac 1 2 L e^{-i\theta_L})\tilde{C}_l^{\prime TT}l^{\prime}e^{i\theta^{\prime}}]+[(\frac 1 2 L e^{i\theta_L})\tilde{C}_l^{\prime TT}l^{\prime}e^{-i\theta^{\prime}}]}$

Each term above can be expressed as $F(\mathbf l)G(\mathbf {l^{\prime}})H(\mathbf L)$.
 * $F(\mathbf{l})= wl[i][0](l)e^{i(s[i][0])\theta_0}$
 * $G(\mathbf{l^{\prime}})= wl[i][1](l^{\prime})e^{i(s[i][1])\theta^{\prime}}$
 * $H(\mathbf{L})= wl[i][2](L)e^{i(s[i][2])\theta_L}$
 
For the 0th (i=0) term ($[(\frac 1 2 L e^{-i\theta_L})\tilde{C}_l^{TT}le^{i\theta_0}]$), decompose it to $F\cdot G\cdot H$, we have:  
* $F = \tilde{C}_l^{TT}le^{i\theta_0}$, $G = -\frac 1 2$, and $H =  L e^{-i\theta_L}$
* Note there is an extra minus sign in the code convention.

Therefore in `class qest_plm_TT(qest)` of `spt3g_software/lensing/lensing/qest.py`
  * $wl[0][0]= \tilde{C}_l^{TT}l$, $s[0][0]=1$  (coefficients for F)
  * $wl[0][1]= -0.5 $, $s[0][1]=0$  (coefficients for G)
  * $wl[0][2]= L$, $s[0][2]=1$  (coefficients for H)

Exercise: fill up other matrix elements of $wl$ and $s$ for TT estimator, and compare to the code.

Snippets from  `spt3g_software/lensing/lensing/qest.py`

`class qest_plm_TT(qest)`
# ```python
class qest_plm_TT(qest):
    '''phi estimated from TT'''
    def __init__(self, cltt):
        self.cltt = cltt
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +1
        self.wl[0][1] = self.wo_d2; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -1
        self.wl[1][1] = self.wo_d2; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wo_d2; self.sl[2][0] = +0
        self.wl[2][1] = self.wc_ml; self.sl[2][1] = +1
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wo_d2; self.sl[3][0] = +0
        self.wl[3][1] = self.wc_ml; self.sl[3][1] = -1
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2

    def wo_d2(self, l, lx, ly):
        return -0.5
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * l

# ```
The method that actually calculates the weighted moments of the two input fields is 
# ```python
    def eval( self, r1, r2, npad=None ):
        '''
        Evaluate the quadradic estimator of \phi from fields r1 and r2.
        '''
        ... 

        for i in xrange(0, self.ntrm):
            term1 = self.wl[i][0](l, lx, ly) * r1.fft * np.exp(+1.j*self.sl[i][0]*psi)
            term2 = self.wl[i][1](l, lx, ly) * r2.fft * np.exp(+1.j*self.sl[i][1]*psi)

            fft[:,:] += (iconvolve_padded(term1, term2, npad=npad)*
                         ( self.wl[i][2](l, lx, ly) *
                           np.exp(-1.j*self.sl[i][2]*psi) ) * 0.5 / np.sqrt(cfft.dx * cfft.dy) * np.sqrt(cfft.nx * cfft.ny))

            #pdb.set_trace()
        return cfft
# ```

It may be confusing, but there is another `qest.py` code (`spt3g_software/lensing/lensing/sims/qest.py`), which is a wrapper of `spt3g_software/lensing/lensing/qest.py`(the one that defines the weight functions, showed above). 

The main library is:
# ```python
class library():
    '''
    Class holding a simulated quadratic estimate of a field
    Initialization inputs: 
      lib_dir,           directory of this library's data products
      cl_unl,            unlensed spectrum
      cl_len,            lensed spectrum
      ivfs1,             cinv-filtered data library for field 1
      ivfs2,             cinv-filtered data library for field 2
      qes,               quadratic estimators.  For example, qes['ptt']=sl.qest.qest_plm_TT
      qfs,               estimator fields
      qrs,               estimator responses
# ```
You can use `get_qft` method to get the weighted moments of the two input fields:
# ```python
    def get_qft(self, k, tft1, eft1, bft1, tft2, eft2, bft2):
        ret = sl.maps.cfft(tft1.nx, tft1.dx, ny=tft1.ny, dy=tft1.dy)
        for tqe, tfe in self.get_qe(k):
            if not isinstance(tqe, tuple):
                ret += self.get_qft( tqe, tft1, eft1, bft1, tft2, eft2, bft2 ) * tfe
            else:
                (qe, f12) = tqe

                # Gets the filtered library FFT corresponding to the requested type
                f1 = {'T' : tft1, 'E' : eft1, 'B' : bft1}[f12[0]]
                f2 = {'T' : tft2, 'E' : eft2, 'B' : bft2}[f12[1]]
                ret += qe.eval(f1, f2) * tfe
        return ret
# ```
`get_sim_qft`(used below) and `get_data_qft` are wrappers of `get_qft`. 

```python
# define the qest object.
# the initialization requires the lensed and unlensed cls, as well as the cinv object.
qest_dd     = sl.sims.qest.library( scan.cl_unl, scan.cl_len, cinv150_len,
                                    lib_dir=bdir+"qest/%s/par_%s/qest_len_dd/"%(ivfs_prefix,qest_prefix) )
```

Compare the reconstructed kappa map with the input kappa map

Note that we need to subtract the mean field bias, which is calculated by `get_sim_qft_mf`

`get_sim_qft_mf` basically runs `get_sim_qft` many times and average over 50 simulations. 

We also need to divide by the response function, which is calculated by  `get_qr`

It basically calculates   $R^{Analytic}_{\mathbf{L}}= \int d^2\mathbf{l}\,W(\mathbf{l}, \mathbf{l^{\prime}})W(\mathbf{l}, \mathbf{l^{\prime}}) F_{\mathbf l }F_{\mathbf {l^{\prime}} }$, which was derived above. 


```python
bl = sl.lensing.spec.bl(60., 3000) # kappa convolved with 1 degree beam
#=== lensed sim plot
k = 'p' # p means minimum variance combination
idx = 1 # do this for the 1st sim
lmax = 2000 # the maximum ell for plotting
# mask with no holes
mask_area = np.load(bdir+"inputs/mask/mask_surv_five_150.npy")
# we need to subtract the mean-field bias from the weighted moments, and then divide the whole thing by
# the normalization factor qest_dd.get_qr
kfft   = (qest_dd.get_sim_qft(k, idx) - qest_dd.get_sim_qft_mf(k, mc_sims_mf))/qest_dd.get_qr(k)*np.arange(0.,lmax+1.)**2/2. * bl
# remove nans in the data
kfft.fft = np.nan_to_num(kfft.fft)
# from fft to map 
kmap   = kfft.get_rffts()[0].get_rmap().map
#=== kappa input plot
# smooth by one degree beam
kfft   = scan.cmblib_proj_len_t1p1.get_sim_kap(idx).get_cfft() * bl
# remove nans
kfft.fft = np.nan_to_num(kfft.fft)
# from fft to map
kmap_in   = kfft.get_rffts()[0].get_rmap().map
#Plot
pl.figure()
pl.subplot(121)
# plot in IAU convention
pl.imshow((mask_area*kmap)[::-1,:], vmin = -.05, vmax = .05)
pl.title('Reconstructed kappa map')
pl.subplot(122)
pl.imshow((mask_area*kmap_in)[::-1,:], vmin = -.05, vmax = .05)
pl.title('Input kappa map')
```
