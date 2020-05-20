---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 0.8.6
  kernelspec:
    display_name: Python 2
    language: python
    name: python2
---

# EXECUTIVE SUMMARY

This notebook fully characterizes the SPT-3G nuller and carrier transfer functions to get accurate noise expectations based on known noise sources and compare to measured noise, measure the bolometer normal resistances, and calculate noise enhancement factors such as from current sharing directly.


This approach is important because the pydfmux warm transfer function in place doesn't consider the cryogenic transfer functions. 

## Nuller Transferfunction

The cryogenic nuller transfer function is **not relevant** for getting the calibration from DAC counts to current through the SQUID when DAN is operating. Consequently **the pydfmux warm transfer function is the correct calibration to use to calibrate readout noise measurements**.

However, individual characterization of the cryogenic portion of this transfer function is required to derive the current sharing effect, whereby noise sources after the summing junction (so in the DEMOD chain) are magnified due to the current division between the squid input coil and the comb. This only applies the noise sources after the summing point while DAN is operating.

## Carrier Transferfunction

The pydfmux carrier transfer function is grossly miscalibrated because it assumes:
* (A) all current out of the SQCB becomes a stiff voltage source due to a single 30mOhm bias resistor
* (B) that the voltage across the bias resistor is the same as the voltage across the bolometer
* (C) that there are no additional paths for current to flow, such as through capacitances to ground in the LC chip that we know exist and can characterize in SPICE.

We can address all three of these points far more accurately, though (C) requires a calibration based on SPICE simulations of the comb and geometric derivations of the expected capacitances. 

The first hint that the carrier transfer function was incorrect came from Rn vs Freq work that Daniel did in 2018 (https://pole.uchicago.edu/spt3g/index.php/Higher_Normal_Resistance_on_select_LC_Channels).

## Data Sources

Transfer function measurements use netanals taken when the stage was just above Tc during the 2018-2019 summer.

Noise measurements are taken from horizon noise stares somewhat later that season.

Normal Resistance measurements are taken from a recent representative drop_bolos pydfmux output file.

## Auxilary Measurements

To understand the noise and the netanal measurements we must know how to convert from a voltage at the ADC to a voltage at the SQUID output.
With SA13s that have substantially higher dynamic impedance, an important component of this becomes the previously unrealized lowpass filter that is the result of the complex impedances in the wiring harness between the SQUID card and the SQCB. 
John Groh first characterized this here (http://www.mcgillcosmology.com/twiki/pub/NextGen_Dfmux/Meetings/2018_Agendas/2018_04_30_readout_meeting_more_wiring_parasitics.pdf)

Rather than use a strict "RC" model like John did, I used measurements of unterminated SA13s provided by Amy Lowitz at ANL to calculate an effective parallel impedance in series with the SQUID output at all frequencies. 
The result of this characterization is imported here, but that work itself is in a separate notebook.

## Results

The results of this indicate that Amy Bender's initial characterization of the SPT-3G readout noise is correct, but that we now understand the source of this `excess noise`.

What we know now is that current sharing is more nuanced than previously modeled, and the effect of the higher dynamic SQUID impedance makes the lowpass filter effect of the wiring harness substantial. When both are considered our dominant noise sources become the voltage and current noise in demodulation chain. 

Under SA4 SQUIDs with <100$\Omega$ dynamic impedance many of these noise sources were ignored, as the resulted in <1pA/rtHz of noise. In the current scenario they contribute as much as 8pA/rtHz at the highest frequencies(!)


# TOC

* Perform imports and pre-load data while defining the resources that will be used
* Detailed description of process
* Code that computes the transfer functions
* Validation of Rn
* Simulation and Validation of Noise

```python
import sys
sys.path.append('/home/joshuam/python')
sys.path.append('/home/joshuam/python/spt3g_software/build/') 
sys.path.append('/home/joshuam/python/spt3g_software/')

import os
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from scipy.constants import Boltzmann  # Used to calculate Johnson noise in sims
from scipy.interpolate import interp1d # Used to build Zeff(f) from the squid lowpass data
from scipy.signal import medfilt       # Used to make easy-to-see representations of noise sources
from pylab import *                    # Yeah. Sue me.

from spt3g import core

import pydfmux
from pydfmux.core.utils.conv_functs import build_hwm_query
from pydfmux.core.utils.transferfunctions import frequency_correction, pi_filter, convert_adc_samples, convert_TF
# This is the transfer function used in the data
# That we need to back out later
pydfmux.set_transferfunction('spt3g_filtering_2017_full')

plt.rcParams.update({
    'font.size': 14,
    'mathtext.fontset': 'cm',
    'text.usetex': True,
    'legend.handlelength': 2,
})
matplotlib.rc('font', **{'family':'sans-serif','sans-serif':['Computer Modern Roman']})
matplotlib.rc('font', **{'family':'serif','serif':['Computer Modern Roman']})
matplotlib.rc('pgf', preamble=[r'\usepackage{siunitx}',r'\sisetup{detect-all}', r"\usepackage[T1]{fontenc}"])
```

## Get Netanal Data used to calculate the current sharing factor (for the nuller TF) and the carrier transfer function

Warm because $R_{bolo}$ matters, and this is pretty close. Ideally we would be using meshed HR10 netanals (or really controlled 450mK netanals) but we don't have these, so this will do.

These are the netanals we have:

[COLD] Lens cap on, Mesh Network analysis, T < Tc (T = 330mk):

`/big_scratch/pydfmux_output/20181209/20181209_041956_take_netanal`     

[WARM] Lens cap on, Mesh network analysis, T > Tc (T = 475 mK and rising):

`/big_scratch/pydfmux_output/20181209/20181209_063609_take_netanal`     

```python
# 475mK warm netanal data (Rbolo~Rn==Rbolo at horizon)
netanal_path = '/big_scratch/pydfmux_output/20181209/20181209_063609_take_netanal/data'

# COLD netanals
#netanal_path = '/big_scratch/pydfmux_output/20181209/20181209_041956_take_netanal/data'
```

```python
netanal_dats = []
for path in os.listdir(netanal_path):
    with open(netanal_path+'/'+path,'rb') as fl:
        dat=pkl.load(fl)
        netanal_dats.append(dat)
```

## Get drop_bolos data to make Rns

Nothing special about this drop_bolos run. 
Just a recent one where we can get the stored normal resistance measurements when the stage is cold and detectors are overbiased.

```python
drop_bolos_path = '/big_scratch/pydfmux_output/20191020/20191020_143154_drop_bolos_371/data/TOTAL_DATA.pkl'
drop_bolos_data = pkl.load(open(drop_bolos_path,'rb'))

# re-arrange the data to be indexed by pstring and only contain useful info
overbias_dat_arrange = {}
for mod in drop_bolos_data:
    if not 'pre_drop' in drop_bolos_data[mod]:
        continue
    for chan in drop_bolos_data[mod]['pre_drop']:
        overbias_dat_arrange[str(mod)+'/'+str(chan)] = drop_bolos_data[mod]['pre_drop'][chan]

# Store the pstrings present in the data. 
# We will use this to only build transfer functions for bolos we have data for
good_ps = overbias_dat_arrange.keys()
```

## Get Horizon Noise Data

```python
d = list(core.G3File('/spt/user/production/calibration/noise/77863968.g3'))
# Similarly, we want the set of bolometers we have transfer functions for to be the same as
# those we have both good noise for, and good normal resistance measurements for.
# Some of the bolos in this measurement have been dropped, so remove any with 0s for noise
good_bnames = [b for b in d[0]["NEI_10.0Hz_to_15.0Hz"].keys() if d[0]["NEI_10.0Hz_to_15.0Hz"][b]>0]
```

## Get SPT-3G pole HWM (valid for the horizon data, drop_bolos data, and netanals)

```python
y = pydfmux.load_session(open('/home/joshuam/python/hardware_maps_southpole/2019/hwm_pole/hwm.yaml','r'))
hwm = y['hardware_map']
```

## Get Effective Parallel Impedance from Wiring Harness

Note that we really only care about this for the DEMOD path, even though it also exists 
in the carrier and nuller path. This is because the effective parallel impedance is, even at the hightest frequencies,
hundreds of ohms. The nuller and carrier paths look like shorts in comparison and no meaningful filter exists.

```python
# See sheet [...] for how I got this. 
# Structure of data is {freqs, Z}, where Z is the effective parallel impedance with the SQUID output
rc_filt_dat = pkl.load(open('/home/joshuam/for_noise_investigation/rc_filter.pkl','rb'))

# "Equivalent impedance from the wireharness that is a parallel impedance with the SQUID output"
Z_par_wh = interp1d(rc_filt_dat['freqs'], rc_filt_dat['Z'])

```

```python
def calc_rc(C, Zd, freqs, L=0):
    """Calculates the effective transfer function of an analytic RCL filter 
    in the wire harness (with a capacitance in parallel with Zdyn and inductance in series)
    
    TekData suggests reasonable numbers are <40pF and <45nH.
    Empirically the best fit is with C=80pF and L=0.
    
    This is still a kind of garbage fit, but is right to ~10%
    
    Divide by the output of this to get the 'enhancement' factor when
    referring voltages at the 1st stage amplifier to the output of the SQUID.
    """
    
    Zl = 1j*2*np.pi*L*freqs
    Zc = -1j*1./(np.pi*2*C*freqs)

    V_out = Zc/(Zd+Zc)
    I_out = V_out / (Zl+10) # 10 ohms in series with the 1st stage amp
    V_measure = I_out * 10  # V at 1st stage amp

    rc_tf = np.abs(V_measure)
    return rc_tf

def calc_rc_fit(freqs, Zd):
    """Empirical calibration for the parallel impedance due to the wire harness.
    
    This uses Amy Lowitz' measurements on unterminated SQUIDs where i've fit for the 
    effective parallel resistance at every frequency to account for the observed transfer function.
    
    This seems portable (on the 4 SQUIDs i've tested it is right to within a few percent and variations
    aren't systematic).
    
    Divide by the output of this to get the 'enhancement' factor when
    referring voltages at the 1st stage amplifier to the output of the SQUID."""
    
    return (Z_par_wh(freqs)/(Zd+Z_par_wh(freqs)))
```

## Pre-compute some DC transfer functions to remove a little tangle later

These are analytic, but trustworthy.
The LT1668 DACs we use produces 10mA (in peak-amplitude). 
Netanals are programmed in "Normalized" amplitudes, from 0-1, which are also peak-amplitudes.

Raw data is in DAN Readout Counts (DROCs), which is a signed 24 bit number.

The first two here convert from normalized and raw units to a current at the DAC output.

```python
# Nuller NORMALIZED amplitude to Amps at DAC output
dac_amps_per_NORMALIZED = 10e-3 # DAC range is 10mA

# DROCs to Amps at DAC OUTPUT
# DAN streamer is 24 bits for full scale
dac_amps_per_DROC = (1/2.**23) * dac_amps_per_NORMALIZED
```

We will also need the scalar difference in current between the carrier and nuller signal paths at the SQCB output.

When we calculate admittance we divide the carrier results by the nuller results. 
This is slick because all of the common transfer function elements (like the Pi Filter and any lowpass filter from the wiring harness) divide out.

However, the two signal paths have different attenuations in an attempt to roughly match their dynamic range at the SQUID (since the carrier sees the comb in parallel with the bias resistor, and the nuller goes directly to the SQUID inputs).

This scalar DC difference needs to be accounted for in the admittance calculation to get the correct units.

The calculation comes from an analytic evaluation of the circuit, and so is subject to component variation, but our QC process does garuantee this to +/-10%.

```python
dc_carrier = ( 300 * ( 200. / 300 ) * ( 1 / 180. ))
dc_nuller = ( 300. * ( 200. / 300. ) * ( 96.77 / ( 100 + 96.77 ) ) / ( 750. * 4 ) )
scalar_difference = dc_carrier/dc_nuller
```

## Parasitic Capacitance Corrections to the Carrier TF
### This is ~irrelevant for the noise analysis, but is important for Rn

These are corrections to the carrier transfer function that
are due to parasitic capacitances in the LC chip.
They come from SPICE sims, but are incomplete.

The true capacitance values should scale for each frequency with
the surface area of the cap. This gives good results at high and 
low frequencies when compared to the data, but is SUPER tedious 
to do in SPICE. For now these are all computed with the capacitances
correct for the lowest frequencies.

You'll see the effect of this in the Rn plots. The estimated Rns
get worse at high frequency because these corrections should be
scaling.

I'll fix this one day when i'm SUPER bored and feel like editing 
265 capacitors by hand in SPICE.



```python
c_corr_spice = np.array([1.        , 0.94748074, 0.88520394, 0.89115315, 0.85535822,
       0.8726872 , 0.85826458, 0.79456523, 0.84047495, 0.80212646,
       0.80975554, 0.83011149, 0.8318509 , 0.83402465, 0.74663681,
       0.75088852, 0.75114929, 0.75034237, 0.74849321, 0.75091757,
       0.75003567, 0.75042027, 1.46564078, 0.73685463, 0.79427805,
       0.72993333, 0.79558679, 0.72345815, 0.73574981, 0.77537633,
       0.79141712, 0.80427878, 0.70652056, 0.78745346, 0.81191846,
       0.68569865, 0.7071563 , 0.79157223, 0.69617845, 0.79103835,
       0.69202064, 0.78647379, 0.68887511, 0.77123836, 0.78078426,
       1.45375977, 0.79463848, 0.80474152, 0.80858794, 0.81485479,
       0.82436464, 0.82956454, 0.84006117, 0.84561185, 0.87284589,
       0.59922124, 0.84576669, 0.88183539, 0.56671881, 0.89423816,
       0.54018127, 0.56284641, 0.56868994, 0.57156128, 0.57305506,
       0.57371985, 0.57409308, 0.57396144])
```

# Detailed Walkthrough of Calculations And Process 

**[You can probably skip this. Comments in code are clearer]**

## NULLER

This is pretty straight forward. Measure the roundtrip transfer function $\frac{I_{SQUID}}{I_{DAC}}$ using a nuller network analysis with the detectors above $T_c$.
Later we can divide our the known pydfmux transfer function to get just the cryogenic portion of this.

This is equivalent to knowing the current sharing factor. This transfer function only applies to signals that
are produced by DAN but don't cancel anything in the synthesizer path (and thus are subject to current division).

## CARRIER

The idea here is that the `true carrier transfer function`
is how to go from current at the DAC output to voltage across the bolometer.

Right now we assume that the carrier and nuller transfer functions are identical up to 
a scaling factor, and we use a combination of analytic DC transfer function calculation
and measurements of the 300K electronics filtering to define the scaling factors and
frequency dependence.

This notebook uses the data from the warm network analysis (so bolos are normal, but LCs are still near base temp)
data to determine a per-channel custom carrier transfer function.

Note that the validity of this transfer function will fall as bolometer resistance falls.
It's still better than what we are doing right now, but shouldn't be taken to be true
when bolometers are in the transition.

This is less straight-forward than characterizing the nuller transfer function because 
the network analysis for the nuller is *directly probing* the quantity of interest, namely
the current through the SQUID for a given DAC output.

The carrier network analysis is probing the current through the SQUID for a given DAC output as well,
but in this case what we actually want is the **voltage across the bolometer** for a given DAC output.

We are only indirectly measuring this, and so we have to make some compromises.

Below I present 3 methods. The 3rd method is the most complete.


## True Circuit

The true circuit we have is the following:

* (A) A perfectly stiff current bias going into the cryogenic circuit.
   * This is a good approximation, the DAC and synthesizer chain can both drive enough current
     not to saturate over the voltage range we deal with.
* (B) A cryogenic circuit ($Z_{TOTAL}$) consisting of three current paths:
   * (1) A very small "bias impedance" in parallel with the rest of the cryogenic circuit.
   This produces a reasonably, but not perfectly, stiff voltage bias to the rest of the circuit.
   We generally assume $Z_{bias}=30m\Omega$, but have also considered the possibility that
   $Z_{bias}=30m\Omega + 2\pi L_{stray}$, such that $L_{stray}\sim1nH$.
   
   * (2) The parallel network of LCR resonators in series with inductive striplines (impedance $Z_{comb}$).
   For the carrier network analysis there is additionally the SQUID input coild
   in series with the network (impedance $Z_{SQUID}$). 
   
   * (3) A set of capacitances to ground between each capacitor and inductor on the LC chips, as well
   as the LC boards themselves and the TES wafers. 
      * These provide the LC-CL asymmetry noted by Daniel.
         We have educated guesses about the capacitances to ground on the TES wafer and LC board, and
         can model the capacitances between the inductor chips to ground and capacitor chips to ground
         as 2D structures using their footprints to estimate that value. Because this capacitance is 
         a function of the area of the individual elements, it has a strong frequency dependence.

We are not able to directly probe (3) with just the data from the network analysis, because its effect is degenerate
with $R_{bolo}$. We will have to construct an approximate correction for this effect based on simulations later.
**For now let's assume no parasitic capacitances.**


Then, the following observations are also true and relevant for the analysis below:

* (A) When DAN is operating the active feedback forces $Z_{SQUID}=0$ at the bias frequencies. Therefore
when the carrier network is taken the series impedance of the SQUID contributes to the circuit, but when
we operate the detectors it does not. 
* (B) When take the Nuller network analysis / Carrier network analysis we are measuring $Z_{comb} + Z_{bias}$,
which I will call ($Z_{DAN}$)
* (C) We currently choose our operating frequencies to be points at which $Z_{DAN}$ is minimized.
For a circuit with no parasitic capacitances this is proveably the point at which the reactance of the striplines + resonator inductors is equal and opposite to the reactance of the resonator capacitances. Therefore, **at our chosen bias frequencies, the input current from the carrier network analysis sees $Z_{bias}$ || $Z_{bolo}+Z_{squid}$, but does NOT see any reactance from the striplines or LCs.**
* (E) And when that same carrier input current is programmed while DAN is operating, that current sees only $Z_{bias}\| Z_{bolo}$.


## Measurement Information

The carrier and the nuller network analyses provide the following:

* $\frac{I_{SQUID}}{I_{DAC}}$ for both the carrier and the nuller.
* We know channels 23 and 46 are $1\Omega$ calibration resistors. So far I have not found this information sufficiently constraining to use it other than as a cross-check...but there may be a way

## Useful Observations

### We can infer $R_{bolo}$ using a linear combination of the nuller and carrier network analyses and some math
* If you assume a perfectly stiff voltage bias, the linear combination (nuller / carrier) network analysis gives us $R_{bolo} + Z_{bias}$ at our chosen bias frequencies. Even with a 1nH $L_{stray}$ at our highest frequencies, $Z_{bias}\sim60m\Omega$. 

* The voltage bias isn't perfectly stiff because it is formed by the bias resistor in parallel with the changing impedance of the SQUID. If we allow ourself to make assumptions about the bias leg impedance and the SQUID we can do the following:

            Comb_R_guess = 1./((1./Z_bias) + (1./(2.+Z_squid)))
            V_comb_corrected = probe_scale * scalar_differance * Comb_R_guess * carrier_netanal

   * The correction is then $V_{corr} = \frac{Z_{bias} \| (R_{bolo} + Z_{SQUID})}{Z_{bias} \| (R_{bolo})}$.
   * For $L_{stray}=0$: $R_{bolo} = 2\Omega$ this is a 0.4% correction at the highest frequencies. For $R_{bolo} = 1\Omega$ this is a 0.7% correction at the highest frequencies.
   * For $L_{stray}=1nH$: $R_{bolo} = 2\Omega$ this is a 1.6% correction at the highest frequencies. For $R_{bolo} = 1\Omega$ this is a 4.2% correction at the highest frequencies.
   
* These are easy to correct for below if we assume the $Z_{SQUID}$ and $L_{stray}$, but note that for a 2 Ohm bolometer these corrections can only be at most $\sim90m\Omega$ off.


## METHOD 1: PyDFMUX-Like
### Assume $Z_{SQUID}<<R_{bolo}$, and $R_{bolo}=\frac{I^{nuller}_{SQUID}}{I^{carrier}_{SQUID}}$, with no correction for $L_{stray}$, and with perfectly stiff CURRENT bias across the comb (rather than across the network).

The carrier network analysis measures $I_{SQUID}$, and we wish to calculate $V_{comb}$.

The simplest assumption we can make is that $I_{SQUID}$ during the carrier network analysis where a voltage bias is applied to both the bolometer and the squid impedance is the same as
$I_{bolo}$ during DAN operation where the squid impedance is nulled away (IE, a current bias). 
Since we can already determine $R_{bolo}$ using a linear combination of carrier and nuller network analyses, it becomes simple to derive $V_{comb}$

We know this is a poor assumption, because we have a largely stiff voltage bias, not a stiff current bias, and $Z_{SQUID}\sim R_{bolo}$ at the highest frequencies (2.03$\Omega$ at 5.4MHz).

Nevertheless, if we do this, then $V_{comb}=I^{carrier}_{SQUID}\times R_{bolo}$.

This is equivalent to $V_{comb} = I^{carrier}_{SQUID} \times \frac{I^{nuller}_{SQUID}}{I^{carrier}_{SQUID}}$.

This reduces to the nuller transfer function, **which is effectively what we currently do with the pydfmux warm transfer function**.

In PyDFMUX these differ by about 5% across the band (but remember that they carry about a +/-10% uncertainty because this is a measured value and QC only ensures individual component variation at the +/-10% level)
### By setting the carrier and nuller transfer functions to be the same (up to some scalar) we reproduce the pydfmux warm transfer function relationship



#### You should now be convinced that the PyDFMUX assumption of the two transfer functions being identical is implicitly an assumption that $Z_{SQUID}<<R_{bolo}$ and should be treated as *wrong*

## METHOD 2: Assume $L_{SQUID}=60nH$ and and perfectly stiff voltage bias. 
### Account for the voltage drop across the SQUID in calculating the expected voltage across the bolometer when DAN is operating.

This method corrects for the fact that the voltage bias is changing as a function of frequency (see Useful Observations above), but still assumes that voltage bias is stiff (IE, $V_{BIAS}$ is constant and will be the same whether or not the SQUID impedance is present).

Method 1 is only measuring part of the voltage drop, since the SQUID is in place and contributes its own impedance.
Carrier netanal: $V_{bolo} = V_{BIAS} - V_{SQUID}$
But when DAN is operating $V_{bolo} = V_{BIAS}$.

So we want to calculate $V_{BIAS} = V_{bolo} + V_{SQUID}$.

This means when DAN is operating, the correct $V_{bolo} = I^{carrier}_{SQUID}\times (R_{bolo}+Z_{SQUID})$

### Commentary

This looks *much* better, in the sense that the apparent voltage bias is no longer strongly attenuated as a function
of frequency due to the SQUID.

Notice that this is going to make the $R_n$ values look much farther off from expectation.
However, this is actually much more in-line with Daniel's expectations based on his SPICE sims (see images 2 & 3 here: https://pole.uchicago.edu/spt3g/index.php/Higher_Normal_Resistance_on_select_LC_Channels)

When he first simulated the comb he was puzzled by the fact that the measured Rns have a strong apparent "dip" relative to his expected corrections. In his wiki he left this as an open question and applied the empirical corrections, making up the differences with some choice normalizations. This is the reason the simulated corrections didn't match the measurements.

### Flaw

**This method still relies on the assumption of a stiff voltage bias.**

It is accounting for the fact that a larger share of that voltage bias is dropped across the SQUID, rather than the Bolometer, and it is also factoring in the fact that the voltage bias is changing as a function of frequency due to the SQUID impedance change. However we are really interested in the voltage across the bolometer when the SQUID is not present, and this doesn't account for an imperfectly stiff voltage bias that depends on the total comb impedance.


## METHOD 3: Calculate $V_{bolometer}$ at every frequency when the SQUID isn't present.

### Can additionally account for residual stripline impedance and parasitic capacitances

            # Calculate total current through the network
            # using what we know about current in one leg
            # and impedances in both of the other legs
            # i_total = (Z_x/Z_total) * I_x
            I_total = i_squid_in_c * (Z_bias + R_bolo + Z_squid + Z_sl_residual)/(Z_bias)

            # we want current through comb when SQUID is not present:
            I_comb_nosquid = (Z_bias / (R_bolo + Z_bias + Z_sl_residual)) * I_total

            # And voltage across bolometer:
            V_bolo = I_comb_nosquid * R_bolo

I've seen no strong motivation for a residual stripline impedance (from not perfectly tuning it out with our choice of frequencies). It's likely to exist at some level, but i've turned this off for now.

I've also applied corrections for the voltage bias attenuation due to current draining through parasitic capacitances to ground, where these are derived from SPICE.

Note that the SPICE sims only use the capacitance values appropriate for the low frequency channels, so we expect some miscalibration vs frequency here as the footprint of the interdigitated capacitors (that form this capacitance to ground) varies. 

I'll get around to fixing this, but it's a lot of caps to change in SPICE.





## Caveats

 * NOTE 1: Biggest uncertainty in all of this comes from the fact that we don't truly know the SQUID dynamic impedances during any of this. They are being approximated using the stored nominal transimpedances. The second biggest uncertainty comes from not knowing the true transimpedances for the noise data. Though i have verified that i expect no degredation during the netanals, so those are probably good approximations for the transfer functions themselves.
 * NOTE 2: 
    The notes for the nuller netanal data say that the stage temperature was drifting a little bit during the 
    measurement. I think this is a small effect (we know the thermal KI effect is small for our LCs),
    and in any case it has no frequency dependence. 

```python
## WARNING: CURRENTLY USES ONLY BOLOS THAT TUNE
def build_transferfunctions(Z_filt=False):
    """Z_filt used to indicate a lowpass filter between the SQUID output and the SQCB input.
    
    Options
    -------
        Z_filt='custom' uses the empirical calibration for effective impedance at every frequency
            derived using Amy Lowitz' measurements.
            
        Z_filt=False assumes no filter
        
        Otherwise it is assumed to be a capacitance in pF and the filter is estimated using
            the analytic calc_rc method above.
            
    Output
    ------
    Dict, see below for structure
    """
    
    custom_dat = {'name' : [],
                  'tf_nuller': [], 
                  'tf_carrier_meth1': [],
                  'tf_carrier_meth2': [],
                  'tf_carrier_meth3': [],
                  'freq' : [],
                  'pstring' : [],
                  'R_bolo' : [],
                  'lc_chan' : [],
                  'squid_zs' : []}

    skipped = 0
    watchdog = 0
    for dat in netanal_dats:
            watchdog+=1
            #print('SQUID: {0}'.format(watchdog))
            if not 'carrier_NA_2nd_pass' in dat: 
                skipped+=1

            # Find SQUID to grab stored nominal Z
            pstring = dat['info']['pathstring_target']
            # some squids have no bolos, skip these
            if not hwm.squids_from_pstring(pstring).count(): 
                continue
            squid = hwm.squids_from_pstring(pstring).one()
            Z = squid.stored_nominal_transimpedance
            if not Z:
                # A few of these don't have stored transimpedances
                # This is because we don't bias them. Skip them.
                skipped+=1
                continue

            # Grab the bolos associated with this SQUID
            bolos = hwm.bolos_from_pstring(dat['info']['pathstring_target']+'/*').filter(pydfmux.Bolometer.tune==True)
            
            # Only keep bolos with good pstrings in drop_bolos data AND noise data
            good_bs = []
            for b in bolos:
                ps = b.pstring()
                name = b.name
                if ps in good_ps and name in good_bnames:
                    good_bs.append(b)
            if not good_bs:
                continue
            bolos = build_hwm_query(good_bs)
            
            # Need the operating bias frequencies
            # the LC channel associated with the detector
            # the bolometer names that used to index the data in the .g3 files
            # and pstrings used natively in pydfmux files
            freqs = np.array([b.lc_channel.frequency for b in bolos])
            lcs = np.array([b.lc_channel.channel for b in bolos])
            bnames = np.array(bolos.name)
            pstrings = np.array(bolos.pstring())

            # Extract the values of the nuller NA and carrier NA
            # Use only the 2nd pass high density data
            n_anal_fs = np.array(dat['nuller_NA_2nd_pass']['freq'])
            c_anal_fs = np.array(dat['carrier_NA_2nd_pass']['freq'])

            # Find the netanal values at the relevant bias frequencies 
            # (this index will be the same for carrier and nuller)
            f_inds = []
            for f in freqs:
                # Don't interpolate, just grab the closest frequency to our bias frequency.
                # These have ~100Hz resolution, so never worse than that. 
                # Typically we're only 5-30Hz off here, which is totally fine.
                f_inds.append((abs(n_anal_fs-f)).argmin())
            f_inds = np.array(f_inds)

            # This uses ONLY data from the 2nd pass of the meshed netanal
            # which uses 2x the programmed probe amplitude. 
            # This is an annoying gotcha and we should fix the data 
            # output info to include the sweep amplitude
            # for both passes to make that clear
            sweep_amp_n = dat['info']['sweep_amplitude_nuller']*2 
            sweep_amp_c = dat['info']['sweep_amplitude_carrier']*2 

            probe_scale = sweep_amp_n/sweep_amp_c

            # sweep_amp is in NORMALIZED units -- need get netanal in AMPS at DAC
            i_dac_n = sweep_amp_n * dac_amps_per_NORMALIZED
            i_dac_c = sweep_amp_c * dac_amps_per_NORMALIZED

            # Convert V_ADC to V_SQCB with known TF
            # The Pi filter correction is ALREADY baked into convert_adc_samples
            # 1.0937500000012903 corrects for the fact that pydfmux is wrong about the
            # amplifier gain
            adc_to_vs = convert_adc_samples('streamer',freqs)
            nuller_v_sqcb = adc_to_vs * np.abs(dat['nuller_NA_2nd_pass']['amp_raw'][f_inds])
            carrier_v_sqcb = adc_to_vs * np.abs(dat['carrier_NA_2nd_pass']['amp_raw'][f_inds])
            
            # netanal gives VOLTS at the input the SQCB.
            # To get to Volts at the SQUID output we apply
            # The correction for the lowpass filter defined by the wiring harness
            if Z_filt:
                if Z_filt=='custom':
                    V_measure = calc_rc_fit(freqs=freqs, Zd=Z+100)
                else:
                    # Assume Z_filt is a capacitance in pF to use
                    # analytic filter calculation
                    V_measure = calc_rc(Z_filt*1e-12, Zd=Z+100, freqs=freqs)
            else: 
                V_measure = 1
            # V_measure converts from V_SQUID to V_1st_stage
            # Divide by this to go from Volts at 1st stage amplifier to Volts at SQUID output
            # to get Amps at SQUID input coil divide by the transimpedance Z 
            i_squid_in_n = (nuller_v_sqcb/V_measure)/Z
            i_squid_in_c = (carrier_v_sqcb/V_measure)/Z
 
            ## NULLER CURRENT SHARING CORRECTION:
            # for the nuller just want H(f) = i_squid_n / i_dac_n 
            h_full_transfer_function = i_squid_in_n/i_dac_n

            ## CARRIER TRANSFER FUNCTIONS (METHODS 1-3)
            
            # SQUID impedance
            Z_squid = 2*np.pi*60e-9*freqs

            # bias_resistor may have a stray L in series with it
            # I have found no strong evidence for this, but will leave it
            # as placeholder
            bias_resistor = 30e-3
            L_bias_stray = 0 #1e-9
            Z_bias = bias_resistor + 2*np.pi*L_bias_stray*freqs     
            
            # Admittance of the comb (removing the SQUID) is
            # Carrier netanal / nuller netanal * difference in probe amplitudes at the DAC * difference
            # in DC transfer functions of warm electronics
            admittance = (probe_scale * (carrier_v_sqcb / nuller_v_sqcb))

            ## METHOD 1: PyDFMUX-like, with carrier tf == nuller tf 
            # (does a better job than pydfmux by correcting for the scalar difference in carrier vs nuller)
            # 
            # FLAWS:
            #     Assumes bias current is the same regardless of SQUID impedance
            #         Which is a current bias, not a voltage bias! 
            #         And therefore fails to account for the voltage drop across the SQUID
            #     Assumes R_bolo = 1/admittance of comb, with no correction for the changing voltage bias
            #         due to SQUID inductance, or accounting for a stray inductance with the bias resistor
            R_bolo = (1./admittance)            
            V_bias_method1 = i_squid_in_c * R_bolo
            
            
            ## METHOD 2: CORRECT FOR SQUID VOLTAGE DROP, and remove bias resistor from Rbolo
            #
            # This assumes a voltage bias that changes vs frequency, 
            # but still assumes a PERFECT voltage bias that won't change when the SQUID impedance
            # is tuned out.
            # First assume R_bolo == 2 ohms and use this to correct for the changing voltage bias
            # (pretty insensitive to choice of R_bolo in the long run. 2 ohms is a fine approximation)
            Comb_R_guess = 1./((1./Z_bias) + (1./(2.+Z_squid)))
            # Correct the inferred admittance based on the changing voltage bias directly
            admittance = (probe_scale * carrier_v_sqcb * scalar_difference * Comb_R_guess) / (nuller_v_sqcb)            
            
            # Admittance actually measures Z_comb + Z_bias. We will assume we are tuning out 
            # all reactances from striplines and LCs, so Z_comb = R_bolo + Z_bias
            R_bolo = (1./admittance) - Z_bias
            # Next, assume a perfectly stiff voltage bias. 
            # So, calculate voltage across the comb in this situation, and assume
            # it will be the same voltage when SQUID is tuned out by DAN
            V_bias_method2 = i_squid_in_c * (R_bolo + Z_squid)

            ## METHOD 3: (1) Correct for imperfect voltage bias where voltage changes 
            #                when SQUID impedance is tuned out
            #            (2) Can also better estimate R_bolo
            #                This is more accurate when the impedance of the TES leg varies from the assumed 2 Ohms
            #                or at higher frequencies, where the SQUID impedance is large.
            #                Typically the correction is only a few percent.
            #            (3) Correct for parasitic capacitances with SPICE sim results
            #
            # Residual Stripline impedance? Matt things maybe a little bit. It's possible. I see
            # no strong evidence.
            Z_sl_residual = 0 #2*np.pi*1e-9*freqs
            # Make assumptions about Z_squid, R_bolo, and Z_bias to guess the effective
            # comb impedance as seen by the nuller
            Comb_R_guess = 1./((1./Z_bias) + (1./(2.+Z_squid+Z_sl_residual)))
            # Correct the inferred admittance based on the changing voltage bias vs frequency
            admittance = (probe_scale * carrier_v_sqcb * scalar_difference * Comb_R_guess) / (nuller_v_sqcb)            
            # Use this to guess at R_bolo based on the effective
            # comb impedance as seen by the nuller with the SQUID removed
            R_bolo = (1./admittance) - Z_bias - Z_sl_residual
            # Calculate total current through the network
            # using what we know about current in one leg
            # and impedances in both of the other legs
            # i_total = (Z_x/Z_total) * I_x
            I_total = i_squid_in_c * (Z_bias + R_bolo + Z_squid + Z_sl_residual)/(Z_bias)

            # we want current through comb when SQUID is not present:
            I_comb_nosquid = (Z_bias / (R_bolo + Z_bias + Z_sl_residual)) * I_total
            
            # this gives us voltage across the comb
            V_bias_method3 = I_comb_nosquid * R_bolo
            
            # Perform empirical correction for the parasitic capacitances from SPICE
            # Based on Daniel's SPICE simulation.
            V_bias_method3 = V_bias_method3  * c_corr_spice[lcs-1]
            
            c_full_transfer_function_meth1 = V_bias_method1 / i_dac_c
            c_full_transfer_function_meth2 = V_bias_method2 / i_dac_c
            c_full_transfer_function_meth3 = V_bias_method3 / i_dac_c

            for n, tf_n, tf_c_1, tf_c_2, tf_c_3, freq, pstring, zs, lc in zip(bnames, 
                                                             h_full_transfer_function, 
                                                             c_full_transfer_function_meth1,
                                                             c_full_transfer_function_meth2,
                                                             c_full_transfer_function_meth3,
                                                             freqs,
                                                             pstrings,
                                                             R_bolo,
                                                             lcs):
                custom_dat['name'].append(n)
                custom_dat['tf_nuller'].append(tf_n) 
                custom_dat['tf_carrier_meth1'].append(tf_c_1)
                custom_dat['tf_carrier_meth2'].append(tf_c_2)
                custom_dat['tf_carrier_meth3'].append(tf_c_3)
                custom_dat['freq'].append(freq)
                custom_dat['pstring'].append(pstring)
                custom_dat['R_bolo'].append(zs)
                custom_dat['lc_chan'].append(lc)
                custom_dat['squid_zs'].append(Z)
                
    for k in custom_dat:
        custom_dat[k] = np.array(custom_dat[k])
    print('Skipped {0} SQUIDs'.format(skipped))
    print('{0} custom TFs stored'.format(len(custom_dat['name'])))
    return custom_dat
```

```python
# "cd"-> "custom_dict"
cd = build_transferfunctions(Z_filt='custom')
```

## Current Sharing Factor

This is the roundtrip nuller transfer function divided by the known pydfmux warm transfer function

```python
%matplotlib notebook

figure(figsize=(12,6))
current_sharing_factor = cd['tf_nuller']*(10e-3/convert_TF(0,'nuller', 'normalized', cd['freq']))

plot(cd['freq']/1e6, 1./current_sharing_factor,
     '.', alpha=0.7,label='Current Sharing Factor')
xlim(1.5,5.5)
#ylim(0,0.07)
title('Factor by which demod chain and SQUID noise is enhanced by "Current Sharing"\nUsing direct measurements of the cryogenic nuller TF')
xlabel('MHz')
ylabel('Enhancement Factor', fontsize=18)
legend(markerscale=2, numpoints=1, loc='best')
grid()
savefig('/home/joshuam/for_noise_investigation/current_sharing_factor.pdf')
```

## Carrier Transfer Function (DAC output to Voltage across *the Bolometer*)

```python
%matplotlib notebook

figure(figsize=(12,6))
plot(cd['freq']/1e6, cd['tf_carrier_meth1'],
     'b.', alpha=0.3,label='METHOD 1 (PyDFMUX Equivalent)')
plot(cd['freq']/1e6, cd['tf_carrier_meth2'],
     'r.', alpha=0.3,label='METHOD 2 (Perfectly Stiff Vbias)')
plot(cd['freq']/1e6, cd['tf_carrier_meth3'],
     'g.', alpha=0.3,label='METHOD 3 (All Known Corrections)')
xlim(1.5,5.5)
ylim(0,0.08)
ylabel(r'$\frac{\mathrm{V}_{Bolo}}{\mathrm{A}_{DAC}}$', rotation=0, fontsize=22, labelpad=30)
xlabel('MHz')
legend(markerscale=3, loc='best', numpoints=1)
grid()
savefig('/home/joshuam/for_noise_investigation/carrier_transfer_functions.pdf')
```

# Use These To Calculate Rns from the cold-overbiased state of a drop-bolos

```python
freqs = []
pydfmux_rns = []
custom_rns_tf1 = []
custom_rns_tf2 = []
custom_rns_tf3 = []
for i in range(len(cd['pstring'])):
    ps = cd['pstring'][i]
    if ps in overbias_dat_arrange.keys():
        # is the bolometer biased?
        if not overbias_dat_arrange[ps]['Cmag']:
            skipped+=1
            continue
        
        # Calculated using the present DFMUX transfer function
        pydfmux_rns.append(overbias_dat_arrange[ps]['R'])
        
        # Get the nuller sorted. Can just use the stored pydfmux ("I"), but i'll be paranoid 
        # and make sure the transfer functions we are using are the same and the import statement at the
        # top didn't get screwed up, or a local pydfmux copy is borked relative to the data copy:
        assert(np.abs((overbias_dat_arrange[ps]['Nmag']* convert_TF(0,'nuller', 'raw', cd['freq'][i])/
               overbias_dat_arrange[ps]['I'])-1)<1e-3)
        
        # And then just use the stored one
        # This is correct because DAN is operating! The cryogenic current division doesn't apply to
        # this signal.
        i_squid_n = overbias_dat_arrange[ps]['I']

        # We do need to use our custom carrier TF to get voltage across the bolometer
        i_dac_c = overbias_dat_arrange[ps]['Cmag'] * dac_amps_per_NORMALIZED
        
        v_comb_1 = i_dac_c * cd['tf_carrier_meth1'][i]
        v_comb_2 = i_dac_c * cd['tf_carrier_meth2'][i]
        v_comb_3 = i_dac_c * cd['tf_carrier_meth3'][i]

        
        r_custom_1 = v_comb_1 / i_squid_n
        r_custom_2 = v_comb_2 / i_squid_n
        r_custom_3 = v_comb_3 / i_squid_n
        
        custom_rns_tf1.append(r_custom_1)
        custom_rns_tf2.append(r_custom_2)
        custom_rns_tf3.append(r_custom_3)
        freqs.append(cd['freq'][i])

freqs = np.array(freqs)
pydfmux_rns = np.array(pydfmux_rns)
custom_rns_tf2 = np.array(custom_rns_tf2) 
custom_rns_tf3 = np.array(custom_rns_tf3)
```

```python
%matplotlib notebook

figure(figsize=(12,6))
plot(freqs/1e6, pydfmux_rns,'.', alpha=0.7,    ms=3, label='pydfmux')
plot(freqs/1e6, custom_rns_tf2,'.', alpha=0.7, ms=3, label='Method 2 (Perfect Voltage Bias)')
#plot(freqs, custom_rns_tf3,'.', alpha=0.7, ms=3, label='Method 3')
ylim(0,5)
grid()
xlabel('MHz')
ylabel('$\Omega$', fontsize=18, rotation=0, labelpad=10)
title("Rns Derived From Previously Used Transfer Functions", fontsize=20)
legend(loc='best', markerscale=5)
savefig('/home/joshuam/for_noise_investigation/old_rn_measurements.pdf')
```

```python
%matplotlib notebook

figure(figsize=(12,6))
plot(freqs/1e6, custom_rns_tf3,'.', alpha=0.9, ms=3, label='Method 3 (All Known Corrections)')
ylim(0,4)
grid()
title('Measured Bolometer Normal Resistance\n(Parasitic capacitance correction uses only low frequency capacitance values)')
xlabel('MHz')
ylabel(r'$\Omega$', rotation=0, fontsize=22, labelpad=15)
legend(loc=1, markerscale=5)
savefig('/home/joshuam/for_noise_investigation/new_rn_measurements.pdf')
```

# Noise (Measurements and Simulations)

Uses known noise sources paired with measurements of:
* The cryogenic current sharing enhancement
* Wiring Harness parallel impedance enhancement
* The carrier transfer function
* SQUID transimpedance measurements (and estimated output impedance values)

## Assumptions

* $Z_{dyn}$ can be estimated from $Z_{trans}$. Note that it seems like they track each other closely, but can vary by about 100 $\Omega$ in either direction. This assumption is a source of uncertainty in the noise estimation.
* $Z_{trans}$ when operating in this configuration is degraded by a factor of 0.75 relative to the stored nominal transimpedance values.
   * This is based on Amy's data on slide 6 here (http://www.mcgillcosmology.com/twiki/pub/NextGen_Dfmux/Meetings/2018_Agendas/LCSQ_update_5_7_2018_ReadoutCall.pdf) where she compares Z degradation vs overbias amplitude. Her measurements start with carriers and DAN operating. I am assuming an additional 5% degradation relative to the pure measurements of Z_trans after SQUID tuning that are stored nominal values.
   * We are quite sensitive to this, so it's worth measuring more completely. This is on deck for testing at ANL.

```python
pydfmux_noise = []
for bolo in cd['name']:
    pydfmux_noise.append(d[0]["NEI_10.0Hz_to_15.0Hz"][str(bolo)] / \
                            (core.G3Units.amp*1e-12 / np.sqrt(core.G3Units.Hz)))
pydfmux_noise = np.array(pydfmux_noise)
```

```python
# Which LCs in LCR vs CLR
# Used to call out the flip-flop in the noise plots.
lcr = [1,2,3,4,5,6,7,9,11,12,13,14,23,25,27,30,31,32,34,35,38,40,42,44,45,46,47,48,49,50,51,52,53,54,55,57,58,60]
```

```python
def calc_noise(cd, scale_zs=1):

    freqs = cd['freq']

    # New tests show that transimpedance doesn't really degrade between tuning, so 
    # this should stay scaled at 1.0
    SQUID_transimpedance = cd['squid_zs']*scale_zs

    # Pi filter correction needed for carrier resistor johnson noise sources
    # that happen after the amplifiers, and so shouldn't get the full TF applied.
    # (this is crossing Ts, but totally irrelevant)
    pi_correction =  pydfmux.core.utils.transferfunctions.pi_filter(cd['freq'])

    # Assume dynamic impedance ~ transimpedance (good to ~100 ohms)
    Zds = cd['squid_zs']*scale_zs*1.1

    # Use the empirical measurement for the equivalent parallel resistance
    # rather than the kinda bullshit "RC" model,
    # DEMOD current noise sources need this actual R_eq to calculate
    # what voltage that current drives across the 1st stage amplifier
    # (Note, the 10 ohm R53 doesn't count towards the total equivalent impedance
    # because C9 and/or ANALOG_VREF sinks the current before it gets there)
    R_eq =  1./((1./Z_par_wh(freqs))+(1./Zds))

    # Voltage noise at the 1st stage amplifier need this
    # TF to get from volts at 1st stage amplifier
    # to volts at SQUID output due to the Z_par_wh
    # and the output impedance
    V_measure = calc_rc_fit(freqs=freqs, Zd=Zds)

    # Cryogenic-only portions of the nuller and carrier transfer functions
    # For the nuller this is the current sharing factor
    # It is the conversion from the current at the SQCB output to
    # a current across the SQUID. Divide by this to get the enhancement 
    # by current sharing
    comb_nuller = cd['tf_nuller']*(10e-3/convert_TF(0,'nuller', 'normalized', cd['freq']))

    # This converts a current at the SQCB output to a voltage across the bolometer
    # Used to convert the SQCB resistor johnson noise to a current noise at the SQUID (need to divide out the 30mohm term)
    # in pydfmux)
    comb_carrier = cd['tf_carrier_meth3']*(10e-3/(convert_TF(0,'carrier', 'normalized', cd['freq'])/0.03))

    # Using Amy's memo
    # CARRIER CONTRIBUTIONS
    cn_dac = 57e-12 * (convert_TF(0,'carrier', 'normalized', cd['freq'])/10e-3)
    cn_amps = 50e-12 * (convert_TF(0,'carrier', 'normalized', cd['freq'])/10e-3)
    # ignore all but the final 4x20 ohm resistors (in practice these are all irrelevant)
    # make them a current noise, turn it into a voltage noise at the bolometer, and then a current noise at the squid
    cn_res = (np.sqrt(4*Boltzmann*300/80) * comb_carrier)/custom_rns_tf3 

    # NULLER CONTRIBUTIONS
    nn_dac = 57e-12 * (convert_TF(0,'nuller', 'normalized', cd['freq'])/10e-3)
    nn_amps = 50e-12 * (convert_TF(0,'nuller', 'normalized', cd['freq'])/10e-3)
    nn_res = 3e-12 * np.ones(len(cd['freq'])) # SQCB 3k resistors = ~2.35pA, 100ohm ~ 0.4, 2x50 ohm mezz ~ 0.2 

    # INTEGRATOR CIRCUIT CONTRIBUTIONS
    r29_john = np.sqrt(4*Boltzmann*300/20e3)* np.ones(len(cd['freq'])) 
    r28_john = np.sqrt(4*Boltzmann*300/20e3)* np.ones(len(cd['freq'])) 

    # CRYOGENIC CONTRIBUTIONS
    bias_resistor = np.sqrt(4*Boltzmann*4*0.03)/custom_rns_tf3 # 30mOhm @ 4K [V/rtHz] / Rbolo
    bolo_johnson = np.sqrt(4*Boltzmann*0.45)/np.sqrt(custom_rns_tf3) # bolos @ 450mK
    squid_noise = 4.5e-12 / comb_nuller # Gets current sharing enhancement

    # DEMOD CONTRIBUTIONS REFERRED TO THE OUTPUT
    # http://dicks-website.eu/noisecalculator/index.html
    # These all gets both the current sharing enhancement and the wire harness filter enhancement
    # Then get divided by transimpedance to get to current at SQUID input /(V_measure*comb_nuller))/SQUID_transimpedance
    # Output - to SQUID input factor:
    ampout_to_isquid = 1./(17.5*V_measure*comb_nuller*SQUID_transimpedance)
    r53_john = (np.sqrt(4*Boltzmann*300*10)*(150./10)) * ampout_to_isquid

    # r59 is basically irrelevant because the current noise gets divided
    # by the 150||10 and we only count the current that goes through the 150
    # feedback resistor
    r59_inoise = (np.sqrt(4*Boltzmann*300./100))
    r59_john = r59_inoise * (10./160)*150 * ampout_to_isquid
    r58_john = (np.sqrt(4*Boltzmann*300*150)) * ampout_to_isquid

    r51_john = ((np.sqrt(4*Boltzmann*300*4.22e3)*17.5)*(Zds/(Zds+4.22e3))) * ampout_to_isquid
    #r51_john = ((np.sqrt(4*Boltzmann*300*4.22e3)*17.5)*(R_eq/(R_eq+4.22e3))) * ampout_to_isquid
    amp_1st_vn = (1.1e-9)*16 * ampout_to_isquid

    # Current noise assumed to be spec for total. Balanced=2.2pA, Unbalanced=3.5pA, 
    # V1
    amp_1st_in = (3.5e-12)*((1./((1./10)+(1./100)+(1./150)))+(1./((1./4.22e3)+(1./Zds))))*17.5* ampout_to_isquid 
    # V2 
    #amp_1st_in = (3.5e-12)*((1./((1./10)+(1./100)+(1./150)))+(1./((1./4.22e3)+(1./R_eq))))*17.5* ampout_to_isquid

    ## 2nd stage amplifier (1.3nV/rtz, 1pa/rthz)
    # <1pA/rtHz at highest freqs 
    stage2 = np.sqrt((1.3e-9)**2+(4*Boltzmann*300*100)+(4*Boltzmann*300*100) +
                     ((4*Boltzmann*300*400)/17.5) + ((4*Boltzmann*300*400)/17.5) +
                     (1e-12*(160./4))**2) * ampout_to_isquid

    tot = np.sqrt(amp_1st_vn**2 + amp_1st_in**2 + 
                  r53_john**2 + r51_john**2 + r59_john**2 + r58_john**2 +
                  squid_noise**2 + bias_resistor**2 + bolo_johnson**2 +
                  nn_dac**2 + nn_res**2 + nn_amps**2 + r28_john**2 + r29_john**2 +
                  cn_dac**2 + cn_amps**2 + cn_res**2+
                  stage2**2)
    
    sources_demod = {'R58 Johnson Noise' : r58_john,
                 '1st Stage Amp Voltage Noise' : amp_1st_vn,
                 'SQUID Noise' : squid_noise,
                 'R59 Johnson Noise' : r59_john,
                 '1st Stage Amp Current Noise' : amp_1st_in, 
                 'R53 Johnson Noise' : r53_john,
                 'R51 Johnson Noise' : r51_john}
    sources_other = {
           'Bias Resistor' : bias_resistor,
           'Bolo Johnson' : bolo_johnson,
           'R28 Johnson Noise' : r28_john,
           'R29 Johnson Noise' : r29_john,
           '2nd Stage Amplification' : stage2}
    sources_nuller = {
           'Nuller (DAC)' : nn_dac,
           'Nuller (Amplifier)' : nn_amps,
           'Nuller (Johnson)' : nn_res}
    sources_carrier = {
           'Carrier (DAC)' : cn_dac,
           'Carrier (Amplifier)' : cn_amps,
           'Carrier (Johnson)' : cn_res}


    return tot, sources_demod, sources_other, sources_nuller, sources_carrier

```

```python
tot, sources_demod, sources_other, sources_nuller, sources_carrier = calc_noise(cd, 1)
```

```python
%matplotlib notebook
tot_0p95 =  calc_noise(cd, 0.95)[0]
tot_0p9 =  calc_noise(cd, 0.9)[0]
tot_0p8 =  calc_noise(cd, 0.8)[0]
lcr_inds = np.array([i in lcr for i in cd['lc_chan']])

figure(figsize=(12,6))

#plot(cd['freq']/1e6,tot*1e12, '.', color='red', alpha=0.2, label='Simulated Noise Expectation')
#plot(cd['freq']/1e6,tot_0p8*1e12, '.', color='b', alpha=0.2, label='Simulated Noise Expectation (20% detuning)')
plot(cd['freq']/1e6,tot_0p95/tot, '.', color='b', alpha=0.5, label='Sim noise improvement (5\% detuning)')
plot(cd['freq']/1e6,tot_0p9/tot, '.', color='c', alpha=0.5, label='Sim noise improvement (10\% detuning)')
plot(cd['freq']/1e6,tot_0p8/tot, '.', color='g', alpha=0.5, label='Sim noise improvement (20\% detuning)')


xlim(1.2,5.5)
legend(numpoints=1)
# handles, labels = plt.gca().get_legend_handles_labels()
# order = [0,1]
# leg=plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=4, markerscale=1)
# for l in leg.get_lines():
#     l.set_alpha(1)
#     l.set_marker('.')
grid(alpha=0.75)
#ylim(0,35)
title("Horizon Noise Data plotted with Approximate Noise Simulation\nSimulation Uses Custom Transfer Functions and SQUID Transimpedances")
xlabel('MHz')
#ylabel(r'$\frac{pA}{\sqrt{Hz}}$', rotation=0, fontsize=20, labelpad=20)
ylabel(r'Readout Noise Ratio', fontsize=20)
savefig('/home/joshuam/for_noise_investigation/horizon_noise_vs_sim.pdf')
```

```python
%matplotlib notebook

lcr_inds = np.array([i in lcr for i in cd['lc_chan']])

figure(figsize=(12,6))
# plot(custom_dat['freq'][np.where(lcr_inds==True)]/1e6,tot[np.where(lcr_inds==True)]*1e12,
#      'r.', alpha=0.2, label='Simulated Expectation')
# plot(custom_dat['freq'][np.where(lcr_inds==False)]/1e6,tot[np.where(lcr_inds==False)]*1e12,
#      '.', color='orange', alpha=0.2, label='Simulated Expectation')

plot(cd['freq'][np.where(pydfmux_noise>0)]/1e6, pydfmux_noise[np.where(pydfmux_noise>0)],
     'b.', alpha=0.2, label='Measured Noise')

plot(cd['freq']/1e6,tot*1e12, '.', color='red', alpha=0.2, label='Simulated Noise Expectation')
# plot(custom_dat['freq'][np.where(lcr_inds==True)]/1e6, 
#      pydfmux_noise[np.where(lcr_inds==True)],'c.', alpha=0.2)#,label='Measured Noise')
# plot(custom_dat['freq'][np.where(lcr_inds==False)]/1e6, 
#      pydfmux_noise[np.where(lcr_inds==False)],'b.', alpha=0.2,label='Measured Noise')

xlim(1.2,5.5)
#legend(markerscale=0, numpoints=1)
handles, labels = plt.gca().get_legend_handles_labels()
order = [0,1]
leg=plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=4, markerscale=1)
for l in leg.get_lines():
    l.set_alpha(1)
    l.set_marker('.')
grid(alpha=0.75)
ylim(0,35)
title("Horizon Noise Data plotted with Approximate Noise Simulation\nSimulation Uses Custom Transfer Functions and SQUID Transimpedances")
xlabel('MHz')
ylabel(r'$\frac{pA}{\sqrt{Hz}}$', rotation=0, fontsize=20, labelpad=20)
savefig('/home/joshuam/for_noise_investigation/horizon_noise_vs_sim.pdf')
```

```python
%matplotlib notebook

arr1inds = freqs.argsort()
freqs_sort = freqs[arr1inds]
excess_noise = pydfmux_noise/(tot*1e12)

figure(figsize=(12,6))
plot(freqs_sort/1e6, excess_noise[arr1inds],'b.', alpha=0.2)
plot(freqs_sort/1e6, medfilt(excess_noise[arr1inds], 501),'r', alpha=0.7, linewidth=2.5,label='Median Filtered')
legend(numpoints=1, markerscale=2)
title("Noise Deviation From Expectation (Ratio)")
grid(alpha=1)
xlabel('MHz')
ylim(0.7,2)
ylabel(r'$\frac{Measurement}{Simulation}$', rotation=0, fontsize=20, labelpad=40)
savefig('/home/joshuam/for_noise_investigation/excess_noise.pdf')
```

```python
%matplotlib notebook

arr1inds = freqs.argsort()
freqs_sort = freqs[arr1inds]
excess_noise = pydfmux_noise-(tot*1e12)

figure(figsize=(12,6))
plot(freqs_sort/1e6, excess_noise[arr1inds],'b.', alpha=0.2)
plot(freqs_sort/1e6, medfilt(excess_noise[arr1inds], 501),'r', alpha=0.7, linewidth=2.5,label='Median Filtered')
legend(numpoints=1, markerscale=2)
title("Excess Noise (Difference between measurement and simulation)")
grid(alpha=1)
xlabel('MHz')
ylim(-5,10)
ylabel(r'$\frac{pA}{\sqrt{Hz}}$', rotation=0, fontsize=20, labelpad=20)
savefig('/home/joshuam/for_noise_investigation/excess_noise_abs.pdf')
```

## Noise Source Breakdown

```python
arr1inds = freqs.argsort()
freqs_sort = freqs[arr1inds]
```

```python
%matplotlib notebook
figure(figsize=(16,8))
for k in ['1st Stage Amp Current Noise', 'SQUID Noise', 'R59 Johnson Noise', '1st Stage Amp Voltage Noise', 
           'R51 Johnson Noise', 'R53 Johnson Noise', 'R58 Johnson Noise']:
    plot(freqs_sort/1e6, 1e12*medfilt(sources_demod[k][arr1inds], 337), '-', linewidth=2, label=k)
for k in sources_nuller:
    plot(freqs_sort/1e6, 1e12*medfilt(sources_nuller[k][arr1inds], 337), '--', linewidth=2, label=k)
for k in sources_carrier:
    plot(freqs_sort/1e6, 1e12*medfilt(sources_carrier[k][arr1inds], 337), ':', linewidth=2, label=k)
for k in sources_other:
    plot(freqs_sort/1e6, 1e12*medfilt(sources_other[k][arr1inds], 337), ':', linewidth=2, label=k)
grid(alpha=0.75)
ylim(0,19)
legend(ncol=3, labelspacing=0.05, numpoints=1, loc='best')
title('Noise Sources In SPT-3G (Simulated)')
xlabel('MHz')
ylabel(r'$\frac{pA}{\sqrt{Hz}}$', rotation=0, fontsize=20, labelpad=20)
savefig('/home/joshuam/for_noise_investigation/simulated_noise_sources.pdf')
```

```python
%matplotlib notebook
figure(figsize=(12,6))

plot(freqs/1e6, 1./calc_rc_fit(freqs=freqs, Zd=cd['squid_zs']*1.1), '.')
grid()
title('Enhancement Factor from ``wiring harness" filter TF')
xlabel('MHz')
ylabel('Enhancement Factor', fontsize=18)
savefig('/home/joshuam/for_noise_investigation/wire_harness_enhancement_factor.pdf')
```

```python

```
