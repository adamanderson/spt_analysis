import numpy as np
import camb
from camb import model
from scipy.interpolate import interp1d
from scipy.integrate import quad, dblquad, simps
from scipy.constants import c
from scipy.special import erf
from numpy.linalg import multi_dot, inv, det


# Set up CAMB
pars = camb.CAMBparams()
pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
pars.InitPower.set_params(ns=0.965)
pars.set_for_lmax(10000, lens_potential_accuracy=0)
pars.NonLinear = model.NonLinear_none
results = camb.get_results(pars)


# Set up CMB TT spectrum
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
totCL = powers['total']
ls = np.arange(totCL.shape[0])
DlTT = interp1d(ls, totCL[:,0])
ClTT = interp1d(ls, totCL[:,0] / (ls*(ls+1)) * (2*np.pi))


# Set up P_eta^perp(k) from Simone
# NB: As defined, Peta_interp below takes arguments that are in units of
# [h / Mpc] and returns a power spectrum that has units of [Mpc^3 / h^3]
Peta_data = np.loadtxt('Peta_data.csv', delimiter=',')
Peta_data = np.vstack([[0,Peta_data[0,1]], Peta_data, [10,0], [100,0]])
Peta_interp = interp1d(Peta_data[:,0], Peta_data[:,1])


# Alternative kSZ models from 1607.01769 (Fig. 1) for validation purposes
# late-time
Dl_ksz_late_sf_data = np.loadtxt('Dl_ksz_late_SF.csv', delimiter=',')
Dl_ksz_late_sf_data = np.vstack([[1, Dl_ksz_late_sf_data[0,1]],
                                 Dl_ksz_late_sf_data,
                                 [10000, Dl_ksz_late_sf_data[-1,1]]])
Dl_ksz_late_sf_interp = interp1d(Dl_ksz_late_sf_data[:,0],
                                 Dl_ksz_late_sf_data[:,1])
ell = Dl_ksz_late_sf_data[:,0]
Cl_ksz_late_sf_interp = interp1d(Dl_ksz_late_sf_data[:,0], 
                                 Dl_ksz_late_sf_data[:,1] / (ell*(ell+1)) * (2*np.pi))

# reionization
Dl_ksz_reion_sf_data = np.loadtxt('Dl_ksz_reion_SF.csv', delimiter=',')
Dl_ksz_reion_sf_data = np.vstack([[1, Dl_ksz_reion_sf_data[0,1]],
                                  Dl_ksz_reion_sf_data,
                                  [10000, Dl_ksz_reion_sf_data[-1,1]]])
ell = Dl_ksz_reion_sf_data[:,0]
Dl_ksz_reion_sf_interp = interp1d(Dl_ksz_reion_sf_data[:,0],
                                  Dl_ksz_reion_sf_data[:,1])
Cl_ksz_reion_sf_interp = interp1d(Dl_ksz_reion_sf_data[:,0],
                                  Dl_ksz_reion_sf_data[:,1] / (ell*(ell+1)) * (2*np.pi))
def Cl_ksz_reion_sf(ell, z_re, delta_z_re):
    return Cl_ksz_reion_sf_interp(ell)


# Late-time kSZ
# This is stolen directly from Shaw, et al. (1109.0553).
# First: source redshift dependence of late-time kSZ ("CSF" curve)
data = np.loadtxt('dDldz_ksz_csf_data.csv', delimiter=',')
data = np.vstack([[0,0], data, [100, data[-1,1]]])
dDlkSZdz_Shaw_interp = interp1d(data[:,0], data[:,1])


# Second: angular spectrum
data = np.sort(np.loadtxt('Dl_ksz.csv', delimiter=','), axis=0)
data = np.vstack([[0,0], data, [10000, data[-1,1]]])
DlkSZ_Shaw_interp = interp1d(data[:,0], data[:,1])


# Derivative of reionization kSZ 2-point function
dlnDl_dlnDeltaz = np.loadtxt('2pt_fisher_deriv_Delta_z.csv', delimiter=',')
dlnDl_dlnDeltaz = np.vstack([[0, dlnDl_dlnDeltaz[0,1]], dlnDl_dlnDeltaz,
                                [10000, dlnDl_dlnDeltaz[-1,1]]])
dlnDl_dlnDeltaz_interp = interp1d(dlnDl_dlnDeltaz[:,0], dlnDl_dlnDeltaz[:,1])
dlnDl_dlntau    = np.loadtxt('2pt_fisher_deriv_tau.csv', delimiter=',')
dlnDl_dlntau    = np.vstack([[0, dlnDl_dlntau[0,1]], dlnDl_dlntau,
                                [10000, dlnDl_dlntau[-1,1]]])
dlnDl_dlntau_interp = interp1d(dlnDl_dlntau[:,0], dlnDl_dlntau[:,1])


class kSZModel:
    def __init__(self, tau, delta_z_re, A_late, alpha_late,
                 beam_fwhm_arcmin, noise_uKarcmin, ell_bin_edges,
                 Cshot, noise_curve_interp=None, use_DlkSZ_reion_derivs=True):
        self.tau = tau
        self.z_re = results.Params.Reion.get_zre(pars, tau=tau)
        self.delta_z_re = delta_z_re
        self.sigma_z_re = delta_z_re / (np.sqrt(8.*np.log(2.)))
        self.A_late = A_late
        self.alpha_late = alpha_late
        self.Dl_3000_ksz_uK2 = 4.0 
        self.beam_fwhm_arcmin = beam_fwhm_arcmin
        self.noise_uKarcmin = noise_uKarcmin
        self.ell_bin_edges = ell_bin_edges
        self.Cshot = Cshot
        self.nbins = len(ell_bin_edges) - 1
        self.noise_curve = noise_curve_interp
        self.use_DlkSZ_reion_derivs = use_DlkSZ_reion_derivs
        
        self.z_interp = np.linspace(0, 15, 100)

        # quantities to pre-compute for speed
        def integrand(z):
            return self.dDlkSZdz_Shaw_notnormed(z)
        self.integral_dDlkSZdz, _ = quad(integrand, 0, 20, epsrel=1e-4)

        self.dKdz_interp = {}
        for component in ['reion', 'late']:
            self.dKdz_interp[component] = {}
            for jbin in np.arange(len(ell_bin_edges)-1):
                self.dKdz_interp[component][jbin] = \
                    interp1d(self.z_interp, [self.dKdz(z, jbin, components=[component]) for z in self.z_interp])


    def Cl_total(self, ell):
        '''
        C_ell noise curve in uK^2

        Parameters
        ----------
        ell : float
            Multipole

        Returns
        -------
        Cl_tot : float (or array)
            Total Cl power including CMB TT, kSZ, and instrumental noise
        '''
        if self.noise_curve is None:
            noise_Cl_uK2_nobeam = (self.noise_uKarcmin * np.pi / 10800)**2
            beam_factor = np.exp((ell**2) * ((self.beam_fwhm_arcmin / np.sqrt(8*np.log(2)) * np.pi / 10800)**2))
            noise_Cl_uK2 = noise_Cl_uK2_nobeam*beam_factor
        else:
            noise_Cl_uK2 = self.noise_curve(ell) * (2*np.pi) / (ell*(ell+1))
        
        Cl_tot = ClTT(ell) + \
                self.Dl_3000_ksz_uK2 / (ell*(ell+1)) * 2*np.pi + noise_Cl_uK2
        return Cl_tot


    def W_s(self, ell, jbin):
        '''
        kSZ filter function

        Parameters
        ----------
        ell : float
            Multipole

        Returns
        -------
        filter : float
            Ell-space filter, given by:
                sqrt(C_l^kSZ) / C_l_total
        '''
        if ell > self.ell_bin_edges[jbin] and ell < self.ell_bin_edges[jbin+1]:
            Cl_tot = self.Cl_total(ell)
            return np.sqrt(self.Dl_3000_ksz_uK2 * 2*np.pi / (ell*(ell+1))) / Cl_tot
        else:
            return 0


    def dDlkSZdz_Shaw_notnormed(self, z):
        return dDlkSZdz_Shaw_interp(z) * 0.5 * (1 - erf((z - self.z_re) / self.sigma_z_re))


    def dDlkSZ3000dz_Shaw(self, z):
        # return self.A_late * DlkSZ_Shaw_interp(3000) * self.dDlkSZdz_Shaw_notnormed(z) / self.integral_dDlkSZdz
        return DlkSZ_Shaw_interp(3000) * self.dDlkSZdz_Shaw_notnormed(z) / self.integral_dDlkSZdz


    def DlkSZ_Shaw(self, ell):
        # return self.A_late * DlkSZ_Shaw_interp(ell) * ((ell / 3000)**self.alpha_late)
        return DlkSZ_Shaw_interp(ell)


    def dCkSZdz_late(self, ell, z, dDlkSZ3000dz=dDlkSZ3000dz_Shaw, DlkSZ=DlkSZ_Shaw):
        '''
        Calculates (d(C_ell^kSZ) / dz) |_late

        Parameters
        ----------
        ell : float
            Multipole
        z : float
            Redshift
        dDlkSZ3000dz : function
            Function that returns d(Dl^kSZ)/dz at ell=3000. Function signature is
            assumed to be of the form:
                dDlkSZ3000dz(z, z_re, delta_z_re)
            where z_re is the mean redshift of reionization, and delta_z_re is the
            25% - 75% duration of reionization
        DlkSZ : function
            Function that returns Dl^kSZ from the late-time effect. Function
            signature is assumed to be of the form:
                DlkSZ(ell)

        Returns
        -------
        dCkSZdz : float
            Result
        '''
        # dDkSZdz = DlkSZ(self, ell) / DlkSZ_Shaw_interp(3000) * dDlkSZ3000dz(self, z) / self.A_late
        dDkSZdz = DlkSZ(self, ell) / DlkSZ_Shaw_interp(3000) * dDlkSZ3000dz(self, z)
        dCkSZdz = dDkSZdz / (ell*(ell+1)) * (2*np.pi)
        return dCkSZdz


    def DlkSZ_reion_flat(self, ell, z_re=None, delta_z_re=None, tau=None,
                         use_derivs=True):
        '''
        Amplitude at ell=3000 of the patchy kSZ power spectrum in D_ell. Amplitude
        is from equation 6 of 2002.06197.

        Parameters
        ----------
        ell : float
            Multipole

        Returns
        -------
        D_ell : float
            D_ell at ell=3000
        '''
        if z_re is None:
            if tau is None:
                z_re = self.z_re
            else:
                # then override z_re given in argument
                z_re = results.Params.Reion.get_zre(pars, tau=tau)
        if delta_z_re is None:
            delta_z_re = self.delta_z_re

        if use_derivs:
            # Method using derivatives
            tau_fid = 0.056
            dtau_dzre_fid = 0.001 / (results.Params.Reion.get_zre(pars, tau=tau_fid+0.001) - results.Params.Reion.get_zre(pars, tau=tau_fid))
            z_re_fid = results.Params.Reion.get_zre(pars, tau=tau_fid)
            delta_z_re_fid = 1.2
            Dl_fiducial = 2.03 * (((1 + z_re_fid) / 11) - 0.12) * (delta_z_re_fid / 1.05)**0.51
            DlkSZ_reion = Dl_fiducial * (1 + dlnDl_dlnDeltaz_interp(ell) * (delta_z_re - delta_z_re_fid) / delta_z_re_fid + \
                                             dlnDl_dlntau_interp(ell) * dtau_dzre_fid * (z_re - z_re_fid) / tau_fid)
        else:
            # Method without derivatives
            DlkSZ_reion = 2.03 * (((1 + z_re) / 11) - 0.12) * (delta_z_re / 1.05)**0.51

        
        return DlkSZ_reion


    def dCkSZdz_reion(self, ell, z, DlkSZ=DlkSZ_reion_flat):
        '''
        Patchy reionization contribution to the kSZ effect. Spectrum in ell is assumed to be flat in D_ell.

        Parameters
        ----------
        ell : float
            Multipole
        z : float
            Redshift
        DlkSZ : function
            Function that returns the reionization Dl's, with signature:
                DlkSZ(ell, z_re, delta_z_re)

        Returns
        -------
        '''
        Cl_kSZ = (DlkSZ(self, ell, use_derivs=self.use_DlkSZ_reion_derivs) / (ell*(ell+1)) * 2 * np.pi)

        norm_pdf = 1./np.sqrt(2*np.pi*(self.sigma_z_re**2)) * \
                np.exp(-1*((z - self.z_re) / self.sigma_z_re)**2 / 2)
        return norm_pdf * Cl_kSZ


    def dKdz(self, z, jbin, components=['reion', 'late']):
        '''
        Calculate dK/dz, the contribution to the K statistic as a function of 
        redshift. This is defined via equation 7 of 1607.01769.

        Parameters
        ----------
        z : float
            Redshift
        component : list of str
            The components to include in the kSZ power. Options include 'reion'
            and 'late'.

        Returns
        -------
        dKdz : float
            dK/dz at redshift z
        '''
        def integrand(ell):
            dCkSZdz = 0
            if 'reion' in components:
                dCkSZdz += self.dCkSZdz_reion(ell, z)
            if 'late' in components:
                dCkSZdz += self.dCkSZdz_late(ell, z)
            if z < 6:   # 1803.07036 uses this arbitrary cutoff for applying the late-time scaling factors
                dCkSZdz *= self.A_late * ((ell / 3000)**self.alpha_late)

            return ell / (2*np.pi) * \
                self.W_s(ell, jbin)**2 * \
                dCkSZdz

        ell_grid = np.linspace(1000,7000,100)
        result = simps([integrand(ell) for ell in ell_grid], ell_grid)
        # result, _ = quad(integrand, 1000, 7000, epsrel=1e-3) #10000)
        return result

    def dCLKKdz(self, z, L, jbin, kbin, components=['reion', 'late'], interp=True):
        '''
        dC_L^KK / dz: contribution to C_L^KK per unit redshift; from equation 
        9 of 1607.01769.

        Parameters
        ----------
        z : float
            Redshift
        L : float
            Multipole
        component : list of str
            The components to include in the kSZ power. Options include 'reion'
            and 'late'.

        Returns
        -------
        dCLKKdz : float
            dC_L^KK / dz
        '''
        dKdz_eval = {}
        for bin in [jbin, kbin]:
            if interp:
                dKdz_eval[bin] = 0
                for component in components:
                    dKdz_eval[bin] += self.dKdz_interp[component][bin](z)
            else:
                dKdz_eval = self.dKdz(z, jbin, components)
            
        little_h = results.hubble_parameter(0) / 100

        # NB: The factor of c/1000 is introduced to cancel the km/s in the Hubble
        # parameter. It is divided by 1000 because `c` from scipy.constants is in
        # m/s by default.
        out = results.hubble_parameter(z) / (c/1000) / \
                (results.comoving_radial_distance(z)**2) * \
                (dKdz_eval[jbin]*dKdz_eval[kbin]) * \
                Peta_interp(L / results.comoving_radial_distance(z) / little_h) / (little_h**3)
        return out


    def Ktotal(self, jbin):
        '''
        Integral of the K field

        Parameters
        ----------
        None

        Returns
        -------
        Ktotal : float
            Integral of K field
        '''
        def integrand(ell):
            Cl_total_ell = self.Cl_total(ell)
            return ell / (2*np.pi) * self.W_s(ell, jbin)**2 * Cl_total_ell
        result, _ = quad(integrand, 100, 7000)
        return result


    def CLKK(self, L, jbin, kbin, components=['reion', 'late'], interp=True,
             shotnoise=False, integrator='quad'):
        '''
        C_L^KK from equation 9 of 1607.01769.

        Parameters
        ----------
        L : float
            Multipole
        component : list of str
            The components to include in the kSZ power. Options include 'reion'
            and 'late'.

        Returns
        -------
        dCLKKdz : float
            dC_L^KK / dz
        '''
        if integrator == 'quad':
            def integrand(z):
                return self.dCLKKdz(z, L, jbin, kbin, components, interp)
            result, _ = quad(integrand, 0.01, 13, epsrel=1e-4)
        elif integrator == 'simps':
            z_eval = np.linspace(0.01, 15, 100)
            dCLKKdz_arr = np.array([self.dCLKKdz(z, L, jbin, kbin, components, interp) for z in z_eval])
            result = simps(dCLKKdz_arr, z_eval)

        
        if shotnoise:
            result += self.Cshot[jbin,kbin]
        
        return result
        

    def NLKK(self, L, jbin, kbin, method='simps'):
        '''
        4-point estimator reconstruction noise

        Parameters
        ----------
        L : float
            Multipole

        Returns
        -------
        NLKK : float
            Noise level of the 4-point estimator
        '''
        if jbin != kbin:
            return 0

        def integrand(theta, ell):
            L_minus_ell = np.sqrt(L**2 + ell**2 - 2.*L*ell*np.cos(theta))
            return 2. / (4*(np.pi**2)) * ell * \
                self.W_s(ell, jbin)**2 * \
                self.W_s(L_minus_ell, jbin)**2 * \
                self.Cl_total(ell) * self.Cl_total(L_minus_ell)
        
        if method == 'quad':
            result, _ = dblquad(integrand, self.ell_bin_edges[jbin], self.ell_bin_edges[jbin+1], 0, 2*np.pi, epsrel=1e-2)
        elif method == 'simps':
            theta = np.linspace(0, 2*np.pi, 50)
            ell = np.linspace(self.ell_bin_edges[jbin], self.ell_bin_edges[jbin+1], 50)
            integ = np.array([[integrand(t, e) for t in theta] for e in ell])
            result = simps(simps(integ, ell), theta)

        return result

    
    def fisher_2point(self, x0, dx, prior, bandpower_errs, bandpower_centers):
        '''
        Calculate Fisher matrix for power spectrum. Reionization kSZ bandpowers
        are assumed to be flat in D_ell.

        Parameters
        ----------
        Arguments `x0`, `dx`, and `prior` describe the model parameters and are
        specified as dictionaries of the form:
            {'tau': tau,
             'delta_z_re': delta_z_re}
        x0 : dict
            Fiducial parameters for forecast.
        dx : dict
            Step-size for parameters.
        prior : dict
            Priors on parameters. Set to 'None' if no prior is desired.
        bandpower_errs : array
            Array of bandpower errors
        bandpower_centers : array
            Array of bandpower centers

        Returns
        -------
        fisher_matrix : array
            Fisher matrix. Matrix is also set to a class variable 
            self.fisher_matrix_2pt
        '''
        x0 = [x0['tau'], x0['delta_z_re']]
        dx = [dx['tau'], dx['delta_z_re']]

        logderivs = [dlnDl_dlntau_interp, dlnDl_dlnDeltaz_interp]

        # compute fisher matrix
        fisher_matrix = np.zeros(shape=(len(x0), len(x0)))
        for ivar in range(len(x0)):
            for jvar in range(len(x0)):
                bandpowers = self.DlkSZ_reion_flat(3000, delta_z_re=x0[1], tau=x0[0], use_derivs=self.use_DlkSZ_reion_derivs) * np.ones(len(bandpower_errs))
                fisher_matrix[ivar, jvar] = np.sum( logderivs[ivar](bandpower_centers) * logderivs[jvar](bandpower_centers) * \
                                                    (bandpowers**2) / (x0[ivar] * x0[jvar]) * \
                                                    (bandpowers**2) / (bandpower_errs**2) )
                
        # apply priors
        if prior['tau'] is not None:
            fisher_matrix[0,0] = fisher_matrix[0,0] + (1./(prior['tau']**2))
        if prior['delta_z_re'] is not None:
            fisher_matrix[1,1] = fisher_matrix[1,1] + (1./(prior['delta_z_re']**2))

        self.fisher_matrix_2pt = fisher_matrix

        return fisher_matrix


class Fisher4Point:
    def __init__(self, fiducial_model, fsky):
        '''
        Constructor for Fisher forecast of 4-point estimator.

        Parameters
        ----------
        fiducial_model : kSZModel object
            Model with fiducial parameters
        '''
        self.fidmodel = fiducial_model
        self.fsky = fsky

    def calc_fisher(self, x0, dx, prior):
        '''
        Parameters
        ----------
        All parameter arguments are dictionaries of the form:
        {'tau': tau,
         'delta_z_re': delta_z_re,
         'A_late': A_late,
         'alpha_late': alpha_late
         'shot_noise': shot_noise (nbins x nbins matrix)}
        '''
        # index 0: tau
        # index 1: duration of reionization
        # Nuisance:
        # index 2: A_late
        # index 3: alpha_late
        # index 4: shot noise component to CLKK

        params_list = ['tau', 'delta_z_re', 'A_late', 'alpha_late']

        # evaluate at 20 points in L
        L_bin_edges = np.logspace(1, np.log10(200), 21)
        L_bin_centers = (L_bin_edges[1:] + L_bin_edges[:-1])/2
        L_bin_widths = L_bin_edges[1:] - L_bin_edges[:-1]

        # check that shot-noise matrix is symmetric
        if np.sum(x0['shot_noise'] - x0['shot_noise'].T) != 0 or \
           np.sum(dx['shot_noise'] - dx['shot_noise'].T) != 0 or \
           (prior['shot_noise'] is not None and \
            np.sum(prior['shot_noise'] - prior['shot_noise'].T) != 0):
            raise ValueError('Shot-noise arugment matrix is not symmetric!')

        
        NLKK_matrix = []
        CLKK_matrix = []
        deriv_CLKK_matrix = {par: [] for par in params_list}
        for jL, L in enumerate(L_bin_centers):
            # calculate noise matrix
            NLKK_matrix.append(np.eye(self.fidmodel.nbins)) 
            for jbin in np.arange(self.fidmodel.nbins):
                NLKK_matrix[jL][jbin,jbin] = self.fidmodel.NLKK(L, jbin, jbin)  

            # calculate CLKK an its derivatives
            CLKK_matrix.append(np.eye(self.fidmodel.nbins))
            for jbin in np.arange(self.fidmodel.nbins):
                for kbin in np.arange(jbin, self.fidmodel.nbins): 
                    CLKK_matrix[jL][jbin, kbin] = self.fidmodel.CLKK(L, jbin, kbin, components=['reion', 'late'],
                                                                     shotnoise=True)
                    CLKK_matrix[jL][kbin, jbin] = CLKK_matrix[jL][jbin, kbin]


        # calculate derivatives of CLKK
        for param in params_list:
            # define a new model at step away from fiducial
            x_step = {par: x0[par] for par in x0}
            x_step[param] += dx[param]

            model_step = kSZModel(tau=x_step['tau'],
                                    delta_z_re=x_step['delta_z_re'],
                                    A_late=x_step['A_late'],
                                    alpha_late=x_step['alpha_late'],
                                    beam_fwhm_arcmin=self.fidmodel.beam_fwhm_arcmin,
                                    noise_uKarcmin=self.fidmodel.noise_uKarcmin,
                                    ell_bin_edges=self.fidmodel.ell_bin_edges,
                                    Cshot=self.fidmodel.Cshot,
                                    noise_curve_interp=self.fidmodel.noise_curve)

            for jL, L in enumerate(L_bin_centers):
                deriv_CLKK_matrix[param].append(np.eye(self.fidmodel.nbins))
                for jbin in np.arange(self.fidmodel.nbins):
                    for kbin in np.arange(jbin, self.fidmodel.nbins): 
                        CLKK_step = model_step.CLKK(L, jbin, kbin, components=['reion', 'late'], shotnoise=True)
                        deriv_CLKK_matrix[param][jL][jbin, kbin] = (CLKK_step - CLKK_matrix[jL][jbin, kbin]) / dx[param]

                        # enforce matrix symmetry
                        deriv_CLKK_matrix[param][jL][kbin, jbin] = deriv_CLKK_matrix[param][jL][jbin, kbin]

        # Calculate the components of the fisher matrix.
        # This requires some massaging of the parameters.
        nparams_shot = int(self.fidmodel.nbins * (self.fidmodel.nbins+1) / 2)
        nparams = 4 + nparams_shot
        fisher_matrix = np.zeros((nparams, nparams))

        # first calculate the 4 x 4 block of the main parameters
        for jparam, param1 in enumerate(params_list):
            for kparam, param2 in enumerate(params_list):
                for jLbin, L in enumerate(L_bin_centers):
                    script_CLKK_matrix = CLKK_matrix[jLbin] + NLKK_matrix[jLbin]
                    CLKK_inv = inv(script_CLKK_matrix)
                    trace_factor = np.trace(multi_dot(arrays = [deriv_CLKK_matrix[param1][jLbin], CLKK_inv,
                                                                deriv_CLKK_matrix[param2][jLbin], CLKK_inv]))
                    fisher_matrix[jparam, kparam] += self.fsky / 2. * (2.*L + 1) * trace_factor * L_bin_widths[jLbin]

        # next, calculate the N*(N+1)/2 x N*(N+1)/2 block for the shot-noise
        # nuisance parameters
        row1 = 0
        col1 = 0
        col_offset1 = 0
        for jparam in np.arange(4, nparams):
            deriv_shot_matrix1 = np.zeros((self.fidmodel.nbins, self.fidmodel.nbins))
            deriv_shot_matrix1[row1, col1] = 1
            deriv_shot_matrix1[col1, row1] = 1
            col1 += 1
            if col1 == self.fidmodel.nbins:
                row1 += 1
                col1 = col_offset1 + 1
                col_offset1 += 1

            row2 = 0
            col2 = 0
            col_offset2 = 0
            for kparam in np.arange(4, nparams):
                deriv_shot_matrix2 = np.zeros((self.fidmodel.nbins, self.fidmodel.nbins))
                deriv_shot_matrix2[row2, col2] = 1
                deriv_shot_matrix2[col2, row2] = 1
                col2 += 1
                if col2 == self.fidmodel.nbins:
                    row2 += 1
                    col2 = col_offset2 + 1
                    col_offset2 += 1

                for jLbin, L in enumerate(L_bin_centers):
                    script_CLKK_matrix = CLKK_matrix[jLbin] + NLKK_matrix[jLbin]
                    CLKK_inv = inv(script_CLKK_matrix)
                    trace_factor = np.trace(multi_dot(arrays = [deriv_shot_matrix1, CLKK_inv,
                                                                deriv_shot_matrix2, CLKK_inv]))
                    fisher_matrix[jparam, kparam] += self.fsky / 2. * (2.*L + 1) * trace_factor * L_bin_widths[jLbin]

        # finally, calculate the 4 x N*(N+1)/2 blocks of cross terms
        for jparam, param1 in enumerate(params_list):
            row2 = 0
            col2 = 0
            col_offset2 = 0
            for kparam in np.arange(4, nparams):
                deriv_shot_matrix2 = np.zeros((self.fidmodel.nbins, self.fidmodel.nbins))
                deriv_shot_matrix2[row2, col2] = 1
                deriv_shot_matrix2[col2, row2] = 1
                col2 += 1
                if col2 == self.fidmodel.nbins:
                    row2 += 1
                    col2 = col_offset2 + 1
                    col_offset2 += 1

                for jLbin, L in enumerate(L_bin_centers):
                    script_CLKK_matrix = CLKK_matrix[jLbin] + NLKK_matrix[jLbin]
                    CLKK_inv = inv(script_CLKK_matrix)
                    trace_factor = np.trace(multi_dot(arrays = [deriv_CLKK_matrix[param1][jLbin], CLKK_inv,
                                                                deriv_shot_matrix2, CLKK_inv]))
                    fisher_matrix[jparam, kparam] += self.fsky / 2. * (2.*L + 1) * trace_factor * L_bin_widths[jLbin]
                    fisher_matrix[kparam, jparam] += self.fsky / 2. * (2.*L + 1) * trace_factor * L_bin_widths[jLbin]
                    
        return NLKK_matrix, CLKK_matrix, deriv_CLKK_matrix, fisher_matrix
            