import numpy as np 
import pickle
import os.path
from scipy.optimize import curve_fit
from scipy.integrate import quad
from pydfmux.analysis import analyze_IV

# physical constants
hPlanck = 6.626e-34 # [J s]
kB = 1.381e-23 # [J / K]

# Load simulated transmission data for the triplexer. Simulations were
# performed by Donna Kubik, and can be obtained on the wiki here:
# https://pole.uchicago.edu/spt3g/index.php/Sonnet_Simulations_related_to_SPT-3G_pixels#08Sep2015_Triplexer_for_Wafer_20.2B
bands_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'triplexer_bands.csv')
bands = np.loadtxt(bands_fname, delimiter=',', skiprows=1)
transmission_freq = bands[:,0]*1e9
transmission = {'90': bands[:,1], '150': bands[:,2], '220': bands[:,3]}


def calc_psat_data(data_files, Zp_data=None, R_term=None, R_threshold='pturn'):
    '''
    Function that extracts the Psat as a function of cold load temperature
    from drop_bolos output files.

    Parameters
    ----------
    data_files : dict
        Python dictionary, with drop_bolos filename, keyed by temperature. In
        the typical application of this function, it is called once per readout
        module.
    Zp_data : dict
        Python dictionary, keyed by bolometer name, with parasitic impedance 
        of each detector.
    R_term : float
        Value of termination resistor on LC board, if present, in Ohm.
    R_threshold : float
        Threshold to use for Psat. If a number between 0 and 1, uses that value
        as the rfrac that corresponds to turnaround (using linear
        interpolation). If 'pturn', then estimates Psat from pturn. Note that
        by construction, the 'pturn' method cannot use linear interpolation
        and therefore results in somewhat noisier Psat estimates.

    Returns
    -------
    PsatVtemp : dict
        Saturation power vs. temperature, keyed by bolometer name.
    '''
    PsatVtemp = dict()
    # loop over temperatures and data to find Psats at each point
    for temp in data_files:
        Psats = dict()
        with open(data_files[temp], 'rb') as f:
            IV_data = pickle.load(f, encoding='latin1')
        Psat_at_T = analyze_IV.find_Psat(data_files[temp], R_threshold=R_threshold,
                                         Zp=Zp_data, R_term=R_term, plot_dir=None)
        
        for channum in Psat_at_T:
            boloname = IV_data['subtargets'][channum]['physical_name']
            if boloname not in PsatVtemp:
                PsatVtemp[boloname] = dict()
                PsatVtemp[boloname]['pstring'] = np.array(IV_data['pstring'])
                PsatVtemp[boloname]['T'] = np.array(list(data_files))
                PsatVtemp[boloname]['Psat'] = np.zeros(len(PsatVtemp[boloname]['T']))
            PsatVtemp[boloname]['Psat'][temp==PsatVtemp[boloname]['T']] = Psat_at_T[channum]

    return PsatVtemp


def PsatofT_coldload(T, Psat_electrical, bias_freq, dark_slope, optical_eff,
                     band, filter_data=None, Rb=0.03, Lb=0.6e-9):
    '''
    Functional form for fitting Psat as a function of coldload temperature in
    order to extract optical efficiency. The model is a blackbody times the
    triplexer transmission function from Donna's Sonnet simulations, and we
    assume that the cold load is beam-filling.

    Parameters
    ----------
    T : float
        Temperature of cold load.
    Psat_electrical : float
        TES saturation power at zero optical loading (i.e. with
        loading from Joule heating only).
    bias_freq : float
        Bias frequency in Hz for doing parasitic bias inductance correction.
    dark_slope : float
        Slope in W/K of the Psat(T_coldload) curve for detectors blanked off
        from the cold load source (either dark or physically blanked).
    optical_eff : float
        Optical efficiency.
    band : str
        Frequency band '90', '150', or '220'
    filter_data : 2-tuple
        Filter transmission data: first entry is frequency of points, second
        entry is the transmission values of the points.
    Rb : float
        Resistance of TES bias (shunt) resistor, in Ohms.
    Lb : float
        Parasitic inductance in series with TES bias (shunt) resistor, in H.

    Returns
    -------
    Psat : float
        Saturation power
    '''
    if band not in ['90', '150', '220']:
        print(band)
        print('WARNING: Invalid band frequency supplied. Defaulting to 220GHz.')
        band = '220'

    def spectral_density(nu, temp):
        if filter_data is None:
            filter_TF = 1.0
        else:
            filter_TF = np.interp(nu, filter_data[0], filter_data[1])
        dPdnu = hPlanck * nu / (np.exp(hPlanck * nu / (kB * temp)) - 1) * \
            np.interp(nu, transmission_freq, transmission[band]) * \
            filter_TF * 1e12
        return dPdnu

    P_optical = np.zeros(len(T))
    for jT in range(len(T)):
        P_optical[jT], _ = quad(spectral_density, a=1e10, b=5e11, args=(T[jT]))
        P_optical[jT] = P_optical[jT] * optical_eff

    Psat = Psat_electrical + dark_slope * T - \
        P_optical/1.e12 * np.sqrt( Rb**2 / (Rb**2 + (2*np.pi*bias_freq*Lb)**2) )
    return Psat


def fit_efficiency(T, Psat, bias_freq, dark_slope, band, filter_data=None,
                   Rb=0.03, Lb=0.6e-9):
    '''
    Fit the efficiency from the saturation power and cold load temperature
    data.

    Parameters
    ----------
    T : array like 
        Temperature of cold load.
    Psat : array like
        Saturation power measured as each cold load temperature.
    bias_freq : float
        Bias frequency in Hz for doing parasitic bias inductance correction.
    dark_slope : float
        Slope in W/K of the Psat(T_coldload) curve for detectors blanked off
        from the cold load source (either dark or physically blanked).
    band : str
        Frequency band '90', '150', or '220'
    filter_data : 2-tuple or None
        Filter transmission data: first entry is frequency of points, second
        entry is the transmission values of the points. None applies no
        correction for filter transmission (transmission = 1).
    Rb : float
        Resistance of TES bias (shunt) resistor, in Ohms.
    Lb : float
        Parasitic inductance in series with TES bias (shunt) resistor, in H.

    Returns
    -------
    model_params : numpy array
        Best-fit parameters of the model. First entry is the saturation power
        at zero optical load, second entry is the efficiency.
    fit_cov : numpy array
        Covariance matrix estimated from `curve_fit` for model parameters.
    '''
    T = np.asarray(T)
    Psat = np.asarray(Psat)

    cNotNormal = Psat > 0.5e-12

    model_params, fit_cov = \
        curve_fit(lambda T, Psat_0, eff: \
                    PsatofT_coldload(T, Psat_0, bias_freq, dark_slope, eff,
                                     band, filter_data=filter_data, Rb=Rb, Lb=Lb),
                    T[cNotNormal], Psat[cNotNormal], p0=[50e-12, 1.2])
    return model_params, fit_cov
