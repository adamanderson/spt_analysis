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

# Some Monte Carlo Studies in Nuclear Gain-Matching
To better understand the effects of the gain-matching procedure.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from spt3g import core
from spt3g.todfilter.polyutils import poly_filter_g3_timestream_map
```

## Statistical errors in nuclear gain-matching
The purpose of this note is to calculate the statistical error in the nuclear gain-matching coefficients. Empirically, we find that we can only gain-match at the level of about 1%. This level could be due to intrinsic polarization of the atmosphere. Or it could be due to statistical error in the nuclear gain-matching coefficient. Let's do a toy Monte Carlo simulation to understand the typical uncertainty of the gain-matching coefficients.

```python
def spectrum_1overf(f, A):
    coeffs = A * (np.abs(f)**-1)
    coeffs[~np.isfinite(coeffs)] = 0
    return coeffs
def spectrum_white(f, w): return w * np.ones(len(f))

def generate_tod(avg_coeffs, freqs):
    abs_freqs = np.abs(freqs)
    sim_fft = np.zeros(len(freqs),dtype=np.complex)
    
    coeffs = np.random.normal(loc=avg_coeffs[freqs>=0],
                              scale=0.1*avg_coeffs[freqs>=0])
    phases = 2*np.pi*np.random.rand(len(coeffs))
    sim_fft[freqs>=0] = coeffs * np.exp(1j*phases)
    sim_fft[(1 + int(len(sim_fft)/2)):] = np.flipud(np.conj(sim_fft[1:int(len(sim_fft)/2):]))
#     for f in freqs>=0:
#         sim_fft[freqs==(-1*f)] = np.conj(sim_fft[freqs==f])
        
        
#     sim_fft[freq<0] = np.conj(sim_fft[])
    
#     sim_fft[freqs==f]
#     for f in freqs[freqs>=0]:
#         avg_coeff = np.unique(avg_coeffs[freqs==f])
#         coeff = np.random.normal(loc=avg_coeff,
#                                  scale=0.1*avg_coeff)
#         phase = 2*np.pi*np.random.rand()
#         sim_fft[freqs==f] = (coeff * np.exp(1j*phase))
#         sim_fft[freqs==(-1*f)] = np.conj(sim_fft[freqs==f])
#     sim_fft = np.asarray(sim_fft)
    
    tod = np.fft.ifft(sim_fft)
    # round off complex error
    tod = np.real(tod)
    
    return tod, sim_fft

def calc_gainmatch_coeffs(tsx, tsy, fmin, fmax):
    fftx = np.fft.fft(tsx)
    ffty = np.fft.fft(tsy)
    freqs = np.fft.fftfreq(len(tsx), d=1./sample_rate)
    f_range = (np.abs(freqs)>fmin) & (np.abs(freqs)<fmax)

    XX = np.sum(np.conj(fftx[f_range])*fftx[f_range])
    YY = np.sum(np.conj(ffty[f_range])*ffty[f_range])
    ReXY = np.sum(np.real(np.conj(fftx[f_range])*ffty[f_range]))
    fX = np.real(np.sqrt( (XX + YY + 2*ReXY) / (2*XX + 2 * ReXY * np.sqrt(XX / YY)) ))
    fY = np.real(np.sqrt( (XX + YY + 2*ReXY) / (2*YY + 2 * ReXY * np.sqrt(YY / XX)) ))
    
    return fX, fY
```

```python
length = 1024*8
sample_rate = 152.5
freqs = np.fft.fftfreq(length, 1./sample_rate)

coeffs_1overf = spectrum_1overf(freqs, 5000)
tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)

coeffs_white1 = spectrum_white(freqs, 5000)
tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

coeffs_white2 = spectrum_white(freqs, 5000)
tod_white2, fft_white1 = generate_tod(coeffs_white2, freqs)
```

```python
plt.figure(figsize=(12,4))
plt.plot(tod_1overf)

plt.figure(figsize=(12,4))
plt.plot(tod_white1)

plt.figure(figsize=(12,4))
plt.plot(tod_white1 + tod_1overf)
```

```python
tsx = tod_1overf + tod_white1
tsy = tod_1overf + tod_white2
```

```python
fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 0.1)
```

```python
ff_sum, psd_sum = periodogram(tsx + tsy, fs=sample_rate)
ff_diff, psd_diff = periodogram(fX*tsx - fY*tsy, fs=sample_rate)
```

```python
_ = plt.loglog(ff_sum, np.sqrt(psd_sum))
_ = plt.loglog(ff_diff, np.sqrt(psd_diff))
plt.ylim([1e-3, 1e3])
```

All the bits basically work, so let's run this over many realizations and estimate the mean leakage and error in the gain-matching coefficients.

```python
def calc_avg_matched_psds(nsims, a_lowf=1000, a_white=500):
    length = 1024*8
    sample_rate = 152.5
    freqs = np.fft.fftfreq(length, 1./sample_rate)
    gainmatch_factors_x = np.zeros(nsims)
    gainmatch_factors_y = np.zeros(nsims)

    avg_psd_sum = np.zeros(int(length/2 + 1))
    avg_psd_diff = np.zeros(int(length/2 + 1))

    for jsim in range(nsims):
        coeffs_1overf = spectrum_1overf(freqs, a_lowf)
        tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)

        coeffs_white1 = spectrum_white(freqs, a_white)
        tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

        coeffs_white2 = spectrum_white(freqs, a_white)
        tod_white2, fft_white2 = generate_tod(coeffs_white2, freqs)

        tsx = np.array(tod_1overf + tod_white1)
        tsx = tsx[:int(length)]
        tsy = np.array(tod_1overf + tod_white2)
        tsy = tsy[:int(length)]
        fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 0.1)
        gainmatch_factors_x[jsim] = fX
        gainmatch_factors_y[jsim] = fY

        ff_sum, psd_sum = periodogram(fX*tsx + fY*tsy, fs=sample_rate)
        ff_diff, psd_diff = periodogram(fX*tsx - fY*tsy, fs=sample_rate)

        avg_psd_sum += psd_sum
        avg_psd_diff += psd_diff

    avg_psd_sum /= nsims
    avg_psd_diff /= nsims
    
    return avg_psd_sum, avg_psd_diff, gainmatch_factors_x, gainmatch_factors_y, ff_sum
```

```python
avg_psd_sum, avg_psd_diff, gainmatch_factors_x, gainmatch_factors_y, freq = \
    calc_avg_matched_psds(nsims=100, a_lowf=1000, a_white=500)
avg_psd_sum2, avg_psd_diff2, gainmatch_factors_x2, gainmatch_factors_y2, freq = \
    calc_avg_matched_psds(nsims=100, a_lowf=10, a_white=500)
```

```python
_ = plt.hist(gainmatch_factors_x, histtype='step')
_ = plt.hist(gainmatch_factors_y, histtype='step')
plt.xlabel('gain-matching coefficients')
plt.title('sigma = {:.3f}'.format(np.std(gainmatch_factors_x)))
plt.tight_layout()
plt.savefig('figures/gain_matching_sim.png', dpi=125)
```

```python
plt.loglog(ff_sum, np.sqrt(avg_psd_sum), 'C0', label='sum (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff), 'C1', label='difference (x-y)')
plt.loglog(ff_sum, np.sqrt(avg_psd_sum2), 'C0--', label='sum (x+y) (1% 1/f power)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff2), 'C1--', label='difference (x-y) (1% 1/f power)')
plt.ylim([1e-1, 1e2])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude spectral density [arb.]')
plt.tight_layout()
plt.legend()
plt.savefig('figures/mean_asd_two_powers.png', dpi=125)
```

The whole point of this section is to demonstrate whether the statistical uncertainty on the gain-matching procedure becomes limiting for this algorithm. To address this problem head on, let's calculate the sum and difference power as a function of the 1/f amplitude.

```python
a_lowf_vals = np.logspace(1, 3, 10)
avg_avg_psd_sum = np.zeros(len(a_lowf_vals))
avg_avg_psd_diff = np.zeros(len(a_lowf_vals))
coeffsx_dict = {}

for jamp, a_lowf in enumerate(a_lowf_vals):
    avg_psd_sum, avg_psd_diff, coeffsx, coeffsy, freq = calc_avg_matched_psds(1000, a_lowf, a_white=500)
    
    avg_avg_psd_sum[jamp] = np.mean(avg_psd_sum[(freq>=0.1) & (freq<0.2)])
    avg_avg_psd_diff[jamp] = np.mean(avg_psd_diff[(freq>=0.1) & (freq<0.2)])
    
    coeffsx_dict[a_lowf] = coeffsx
```

```python
plt.semilogx(avg_avg_psd_sum, avg_avg_psd_diff, 'o')
plt.xlabel('average power in x+y from 0.1 to 0.2 Hz [arb.]')
plt.ylabel('average power in x-y from 0.1 to 0.2 Hz [arb.]')
plt.tight_layout()
plt.savefig('figures/sum_vs_diff_power_vs_lowf_power')
```

```python
for ja, a in enumerate(coeffsx_dict):
    col = plt.cm.viridis(ja / len(coeffsx_dict))    
    plt.hist(coeffsx_dict[a], bins=np.linspace(0.7, 1.3, 41),
             histtype='step', color=col)
plt.xlabel('gain-matching coeff')
plt.tight_layout()
plt.savefig('figures/gain_matching_coeffs_hist.png')
```

## Interaction of poly filter with nuclear gain-matching
Both the polynomial filter and the nuclear gain-matching are methods of eliminating low-frequency noise. In our simulation model, we have assumed that all 1/f noise is 100% in common between both detectors in a polarization pair, so it is no surprise that it is removed with perfect efficiency by the nuclear gain-matching. What happens when we also apply a polynomial filter, as we do in the real data. One might imagine that the noise actually becomes worse because the polynomial subtraction will be subject to statistical fluctuations in each of the two detectors' timestreams. We can easily simulate this as well.

```python
ts_map = core.G3TimestreamMap()
ts_map['x'] = core.G3Timestream(tsx)
ts_map['y'] = core.G3Timestream(tsy)
ts_polyfiltered = poly_filter_g3_timestream_map( ts_map, 10 )
```

```python
plt.plot(ts_polyfiltered['x'])
plt.plot(ts_polyfiltered['y'])
```

```python
nsims = 1000
length = 1024*8
sample_rate = 152.5
freqs = np.fft.fftfreq(length, 1./sample_rate)
time = np.arange(length) / sample_rate

avg_psd_sum = np.zeros(int(length/2 + 1))
avg_psd_diff = np.zeros(int(length/2 + 1))
avg_psd_sum_poly = np.zeros(int(length/2 + 1))
avg_psd_diff_poly = np.zeros(int(length/2 + 1))

for jsim in range(nsims):
    coeffs_1overf = spectrum_1overf(freqs, 1000)
    tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)

    coeffs_white1 = spectrum_white(freqs, 500)
    tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

    coeffs_white2 = spectrum_white(freqs, 500)
    tod_white2, fft_white2 = generate_tod(coeffs_white2, freqs)
    
    tsx = tod_1overf + tod_white1
    tsy = tod_1overf + tod_white2
    
    ts_map = core.G3TimestreamMap()
    ts_map['x'] = core.G3Timestream(tsx)
    ts_map['y'] = core.G3Timestream(tsy)
    ts_polyfiltered = poly_filter_g3_timestream_map( ts_map, 4 )
    
    fX_poly, fY_poly = calc_gainmatch_coeffs(ts_polyfiltered['x'],
                                             ts_polyfiltered['y'], 0.01, 0.1)
    fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 0.1)
    
    ff_sum_poly, psd_sum_poly = periodogram(fX_poly*ts_polyfiltered['x'] + \
                                            fY_poly*ts_polyfiltered['y'],
                                            fs=sample_rate)
    ff_diff_poly, psd_diff_poly = periodogram(fX_poly*ts_polyfiltered['x'] - \
                                              fY_poly*ts_polyfiltered['y'],
                                              fs=sample_rate)
    avg_psd_sum_poly += psd_sum_poly
    avg_psd_diff_poly += psd_diff_poly
    
    ff_sum, psd_sum = periodogram(fX*tsx + fY*tsy, fs=sample_rate)
    ff_diff, psd_diff = periodogram(fX*tsx - fY*tsy, fs=sample_rate)
    avg_psd_sum += psd_sum
    avg_psd_diff += psd_diff
    
avg_psd_sum_poly /= nsims
avg_psd_diff_poly /= nsims
avg_psd_sum /= nsims
avg_psd_diff /= nsims
```

```python
plt.loglog(ff_sum, np.sqrt(avg_psd_sum), 'C0--', label='no poly (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff), 'C1--', label='no poly (x-y)')
plt.loglog(ff_sum, np.sqrt(avg_psd_sum_poly), 'C0', label='with poly (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff_poly), 'C1', label='with poly (x-y)')
plt.ylim([1e-1, 1e3])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude spectral density [arb.]')
plt.legend()
plt.tight_layout()
plt.savefig('figures/mean_asd_poly.png', dpi=125)
```

### Including other sources of 1/f power with different gains


There are at least four major low-frequency noise sources in our detectors:

* atmosphere
* thermal fluctuations on the focal plane
* readout noise, specifically the DAC chain which dominates at low frequencies
* gain drifts in the electronics

In general, atmosphere dominates the low-frequency spectrum. Unfortunately these other sources do not have the same relative calibration between x and y in general. Let's model this w.l.o.g. as a polynomial added to one of the timestreams.

```python
#poly = np.polyval((np.random.rand(5)-1/2), time/100)
zeros = np.random.rand(4)
time_norm = np.linspace(0,1,len(tsx))
poly = (time_norm - zeros[0]) * (time_norm - zeros[1]) * (time_norm - zeros[2]) * (time_norm - zeros[3])
poly_rescaled = 3*poly*np.std(tsx)/(np.max(poly) - np.min(poly))
poly_rescaled -= np.mean(poly_rescaled)

plt.plot(tsx, label='white noise + correlated 1/f (y)')
plt.plot(tsx + poly_rescaled, label='white noise + correlated 1/f + random 4th-order polynomial (x)')
plt.plot(poly_rescaled, label='random 4th-order polynomial')
plt.legend()
plt.tight_layout()
plt.savefig('figures/random_poly_tod.png')
```

```python
nsims = 1000
length = 1024*8
sample_rate = 152.5
freqs = np.fft.fftfreq(length, 1./sample_rate)
time = np.arange(length) / sample_rate

avg_psd_sum = np.zeros(int(length/2 + 1))
avg_psd_diff = np.zeros(int(length/2 + 1))
avg_psd_sum_poly = np.zeros(int(length/2 + 1))
avg_psd_diff_poly = np.zeros(int(length/2 + 1))

for jsim in range(nsims):
    # generate white and 1/f noise
    coeffs_1overf = spectrum_1overf(freqs, 1000)
    tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)

    coeffs_white1 = spectrum_white(freqs, 500)
    tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

    coeffs_white2 = spectrum_white(freqs, 500)
    tod_white2, fft_white2 = generate_tod(coeffs_white2, freqs)
    
    # generate TOD
    tsx = tod_1overf + tod_white1
    tsy = tod_1overf + tod_white2
    
    # generate polynomial to add to one polarization
    zeros = np.random.rand(4)
    time_norm = np.linspace(0,1,len(tsx))
    poly = (time_norm - zeros[0]) * (time_norm - zeros[1]) * (time_norm - zeros[2]) * (time_norm - zeros[3])
    poly_rescaled = 3*poly*np.std(tsx)/(np.max(poly) - np.min(poly))
    poly_rescaled -= np.mean(poly_rescaled)
    
#     poly = np.polyval((np.random.rand(5)-1/2), time/100)
#     poly_rescaled = poly*np.std(tsx)/(np.max(poly) - np.min(poly))
#     poly_rescaled -= np.mean(poly_rescaled)
    
    # add poly to one polarization
    tsx += poly_rescaled
    
    # poly filter, gain match and difference
    ts_map = core.G3TimestreamMap()
    ts_map['x'] = core.G3Timestream(tsx)
    ts_map['y'] = core.G3Timestream(tsy)
    ts_polyfiltered = poly_filter_g3_timestream_map( ts_map, 4 )
    
    fX_poly, fY_poly = calc_gainmatch_coeffs(ts_polyfiltered['x'],
                                             ts_polyfiltered['y'], 0.01, 0.1)
    fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 0.1)
    
    ff_sum_poly, psd_sum_poly = periodogram(fX_poly*ts_polyfiltered['x'] + \
                                            fY_poly*ts_polyfiltered['y'],
                                            fs=sample_rate)
    ff_diff_poly, psd_diff_poly = periodogram(fX_poly*ts_polyfiltered['x'] - \
                                              fY_poly*ts_polyfiltered['y'],
                                              fs=sample_rate)
    avg_psd_sum_poly += psd_sum_poly
    avg_psd_diff_poly += psd_diff_poly
    
    ff_sum, psd_sum = periodogram(fX*tsx + fY*tsy, fs=sample_rate, window='hanning')
    ff_diff, psd_diff = periodogram(fX*tsx - fY*tsy, fs=sample_rate, window='hanning')
    avg_psd_sum += psd_sum
    avg_psd_diff += psd_diff
    
avg_psd_sum_poly /= nsims
avg_psd_diff_poly /= nsims
avg_psd_sum /= nsims
avg_psd_diff /= nsims
```

```python
plt.loglog(ff_sum, np.sqrt(avg_psd_sum), 'C0--', label='no poly filter (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff), 'C1--', label='no poly filter (x-y)')
plt.loglog(ff_sum, np.sqrt(avg_psd_sum_poly), 'C0', label='with poly filter (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff_poly), 'C1', label='with poly filter (x-y)')
plt.ylim([1e-1, 1e3])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude spectral density [arb.]')
plt.title('X = white noise + common 1/f + random 4th-order polynomial\n'
          'Y = white noise + common 1/f')
plt.legend()
plt.tight_layout()
plt.savefig('figures/mean_asd_polyfilter_polysignalX.png', dpi=125)
```

Not surprisingly, the poly filter completely annihilates the polynomial. It removes polynomial power at even higher frequencies than the frequency at which it compromises the white noise, likely as a result of the difference between sinusoidal and polynomial basis functions.

As a further permutation on this test, let's add in an additional 1/f-like component to X instead of the polynomial, to see what happens.

```python
nsims = 1000
length = 1024*8
sample_rate = 152.5
freqs = np.fft.fftfreq(length, 1./sample_rate)
time = np.arange(length) / sample_rate

avg_psd_sum = np.zeros(int(length/2 + 1))
avg_psd_diff = np.zeros(int(length/2 + 1))
avg_psd_sum_poly = np.zeros(int(length/2 + 1))
avg_psd_diff_poly = np.zeros(int(length/2 + 1))

for jsim in range(nsims):
    # generate white and 1/f noise
    coeffs_1overf = spectrum_1overf(freqs, 1000)
    tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)
    
    coeffs_1overf_X = spectrum_1overf(freqs, 200)
    tod_1overf_X, fft_1overf_X = generate_tod(coeffs_1overf_X, freqs)

    coeffs_white1 = spectrum_white(freqs, 500)
    tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

    coeffs_white2 = spectrum_white(freqs, 500)
    tod_white2, fft_white2 = generate_tod(coeffs_white2, freqs)
    
    # generate TOD
    tsx = tod_1overf + tod_white1 + tod_1overf_X
    tsy = tod_1overf + tod_white2
    
    # poly filter, gain match and difference
    ts_map = core.G3TimestreamMap()
    ts_map['x'] = core.G3Timestream(tsx)
    ts_map['y'] = core.G3Timestream(tsy)
    ts_polyfiltered = poly_filter_g3_timestream_map( ts_map, 4 )
    
    fX_poly, fY_poly = calc_gainmatch_coeffs(ts_polyfiltered['x'],
                                             ts_polyfiltered['y'], 0.01, 0.1)
    fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 0.1)
    
    ff_sum_poly, psd_sum_poly = periodogram(fX_poly*ts_polyfiltered['x'] + \
                                            fY_poly*ts_polyfiltered['y'],
                                            fs=sample_rate)
    ff_diff_poly, psd_diff_poly = periodogram(fX_poly*ts_polyfiltered['x'] - \
                                              fY_poly*ts_polyfiltered['y'],
                                              fs=sample_rate)
    avg_psd_sum_poly += psd_sum_poly
    avg_psd_diff_poly += psd_diff_poly
    
    ff_sum, psd_sum = periodogram(fX*tsx + fY*tsy, fs=sample_rate, window='hanning')
    ff_diff, psd_diff = periodogram(fX*tsx - fY*tsy, fs=sample_rate, window='hanning')
    avg_psd_sum += psd_sum
    avg_psd_diff += psd_diff
    
avg_psd_sum_poly /= nsims
avg_psd_diff_poly /= nsims
avg_psd_sum /= nsims
avg_psd_diff /= nsims
```

```python
plt.loglog(ff_sum, np.sqrt(avg_psd_sum), 'C0--', label='no poly (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff), 'C1--', label='no poly (x-y)')
plt.loglog(ff_sum, np.sqrt(avg_psd_sum_poly), 'C0', label='with poly (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff_poly), 'C1', label='with poly (x-y)')
plt.ylim([1e-1, 1e3])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude spectral density [arb.]')
plt.title('X = white noise + common 1/f + X-only 1/f\n'
          'Y = white noise + common 1/f')
plt.legend()
plt.tight_layout()
# plt.savefig('figures/mean_asd_polyfilter_polysignalX.png', dpi=125)
```

### Negative bias from nuclear gain-matching
Another concern that one might have is that nuclear gain-matching results in reduced signal, with the algorithm using noise fluctuations to cancel signal. In the case of a large (e.g. 30% in the test below) gain-mismatches, there is essentially no loss of signal except in the lowest bin.

```python
nsims = 1000
length = 1024*8
sample_rate = 152.5
freqs = np.fft.fftfreq(length, 1./sample_rate)
time = np.arange(length) / sample_rate

avg_psd_sum = np.zeros(int(length/2 + 1))
avg_psd_diff = np.zeros(int(length/2 + 1))
avg_psd_sum_matched = np.zeros(int(length/2 + 1))
avg_psd_diff_matched = np.zeros(int(length/2 + 1))
avg_psd_signal = np.zeros(int(length/2 + 1))

gain_mismatch = 0.3

for jsim in range(nsims):
    # generate white and 1/f noise
    coeffs_1overf = spectrum_1overf(freqs, 1000)
    tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)
    
    coeffs_1overf_X = spectrum_1overf(freqs, 200)
    tod_1overf_X, fft_1overf_X = generate_tod(coeffs_1overf_X, freqs)

    coeffs_white1 = spectrum_white(freqs, 500)
    tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

    coeffs_white2 = spectrum_white(freqs, 500)
    tod_white2, fft_white2 = generate_tod(coeffs_white2, freqs)
    
    # generate TOD
    tsx = (1+gain_mismatch/2)*(tod_1overf + tod_white1 + tod_1overf_X)
    tsy = (1-gain_mismatch/2)*(tod_1overf + tod_white2)
    
    fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 1.0)
    
    ff_sum, psd_sum = periodogram(tsx + tsy, fs=sample_rate, window='hanning')
    ff_diff, psd_diff = periodogram(tsx - tsy, fs=sample_rate, window='hanning')
    
    ff_sum_matched, psd_sum_matched = periodogram(fX*tsx + fY*tsy, fs=sample_rate, window='hanning')
    ff_diff_matched, psd_diff_matched = periodogram(fX*tsx - fY*tsy, fs=sample_rate, window='hanning')
    
    ts_signal = (1+gain_mismatch/2)*(tod_1overf_X + tod_white1) + \
                (1-gain_mismatch/2)*tod_white2
    ff_diff_signal, psd_diff_signal = periodogram(ts_signal, fs=sample_rate, window='hanning')
    
    avg_psd_sum += psd_sum
    avg_psd_diff += psd_diff
    avg_psd_sum_matched += psd_sum_matched
    avg_psd_diff_matched += psd_diff_matched
    avg_psd_signal += psd_diff_signal
    
avg_psd_sum /= nsims
avg_psd_diff /= nsims
avg_psd_sum_matched /= nsims
avg_psd_diff_matched /= nsims
avg_psd_signal /= nsims
```

```python
plt.loglog(ff_sum, np.sqrt(avg_psd_sum), 'C0', label='no matching, 30% gain-mismatch (x+y)')
plt.loglog(ff_diff, np.sqrt(avg_psd_diff), 'C1', label='no matching, 30% gain-mismatch (x-y)')
plt.loglog(ff_sum_matched, np.sqrt(avg_psd_sum_matched), 'C0--', label='gain-matched (x+y)')
plt.loglog(ff_diff_matched, np.sqrt(avg_psd_diff_matched), 'C1--', label='gain-matched (x-y)')
plt.loglog(ff_diff_matched, np.sqrt(avg_psd_signal), 'C2', label='no common 1/f component (x-y)')
plt.ylim([1e-1, 1e3])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude spectral density [arb.]')
plt.title('X = 1.15 $\\times$ (white noise + common 1/f + X-only 1/f "signal")\n'
          'Y = 0.85 $\\times$ (white noise + common 1/f)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/mean_asd_biascheck.png', dpi=125)
```

## Averaging before PSD
The simulations above are a little unfair in that they average the PSDs of the timestreams rather than averaging the timestream and taking the PSD of the average.

```python
nsims = 1000
length = 1024*8
sample_rate = 152.5
freqs = np.fft.fftfreq(length, 1./sample_rate)
time = np.arange(length) / sample_rate

avg_tod_sum = np.zeros(length)
avg_tod_diff = np.zeros(length)
avg_tod_sum_matched = np.zeros(length)
avg_tod_diff_matched = np.zeros(length)

avg_gain_mismatch = 0.1

for jsim in range(nsims):
    # generate white and 1/f noise
    coeffs_1overf = spectrum_1overf(freqs, 1000)
    tod_1overf, fft_1overf = generate_tod(coeffs_1overf, freqs)
    
    coeffs_white1 = spectrum_white(freqs, 500)
    tod_white1, fft_white1 = generate_tod(coeffs_white1, freqs)

    coeffs_white2 = spectrum_white(freqs, 500)
    tod_white2, fft_white2 = generate_tod(coeffs_white2, freqs)
    
    # generate TOD
    gain_mismatch = np.random.normal(loc=0, scale=avg_gain_mismatch)
    tsx = ((1+gain_mismatch/2)*tod_1overf + tod_white1)
    tsy = ((1-gain_mismatch/2)*tod_1overf + tod_white2)
    
    fX, fY = calc_gainmatch_coeffs(tsx, tsy, 0.01, 1.0)
    
    avg_tod_sum += (tsx + tsy)
    avg_tod_diff += (tsx - tsy)
    avg_tod_sum_matched += (fX*tsx + fY*tsy)
    avg_tod_diff_matched += (fX*tsx - fY*tsy)
    
avg_tod_sum /= nsims
avg_tod_diff /= nsims
avg_tod_sum_matched /= nsims
avg_tod_diff_matched /= nsims

ff_sum, psd_sum = periodogram(avg_tod_sum, fs=sample_rate, window='hanning')
ff_diff, psd_diff = periodogram(avg_tod_diff, fs=sample_rate, window='hanning')

ff_sum_matched, psd_sum_matched = periodogram(avg_tod_sum_matched,
                                              fs=sample_rate, window='hanning')
ff_diff_matched, psd_diff_matched = periodogram(avg_tod_diff_matched,
                                                fs=sample_rate, window='hanning')
```

```python
plt.loglog(ff_sum, np.sqrt(psd_sum), 'C0', label='no matching, 10% gain-mismatch (x+y)')
plt.loglog(ff_diff, np.sqrt(psd_diff), 'C1', label='no matching, 10% gain-mismatch (x-y)')
plt.loglog(ff_sum_matched, np.sqrt(psd_sum_matched), 'C0--', label='gain-matched (x+y)')
plt.loglog(ff_diff_matched, np.sqrt(psd_diff_matched), 'C1--', label='gain-matched (x-y)')
plt.xlim([1e-1, 1])
plt.grid()
plt.xlabel('frequency [Hz]')
plt.ylabel('amplitude spectral density [arb.]')
# plt.title('X = 1.15 $\\times$ (white noise + common 1/f + X-only 1/f "signal")\n'
#           'Y = 0.85 $\\times$ (white noise + common 1/f)')
plt.legend()
plt.tight_layout()
plt.savefig('figures/mean_asd_biascheck.png', dpi=125)
```

```python
plt.plot(avg_tod_diff)
plt.plot(avg_tod_diff_matched)
```

```python

```
