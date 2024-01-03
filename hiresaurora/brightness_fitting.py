import astropy.units as u
import numpy as np
from lmfit.models import GaussianModel, ConstantModel
from lmfit.parameter import Parameters
from astropy.convolution import Gaussian1DKernel

from hiresaurora.general import _emission_lines
from hiresaurora.masking import _Mask


class _Spectrum1D:

    def __init__(self, target_line: str, data: u.Quantity, unc: u.Quantity,
                 shifted_wavelengths: u.Quantity, masks: _Mask,
                 target_width_bins: float, slit_width_bins: float):
        self._emission_line = _emission_lines[target_line]
        self._data = data
        self._unc = unc
        self._wavelengths = shifted_wavelengths
        self._masks = masks
        self._target_width = target_width_bins
        self._slit_width = slit_width_bins
        self._kernel = Gaussian1DKernel(stddev=1)
        self._1d_spectra = self._get_1d_spectra()
        self._best_fit, self._best_fit_unc, self._width = self._get_best_fit()
        self._brightness, self._brightness_unc = \
            self._get_integrated_brightness()
        
    def _get_data_slice(self) -> (np.s_, np.s_):
        profile = np.sum(self._masks.target_mask, axis=1)
        target_ind = np.where(np.isnan(profile))[0]
        target = np.s_[target_ind]
        n_target = len(target_ind)
        return target, n_target
    
    def _get_1d_spectra(self) -> dict:
        target_slice, n_target = self._get_data_slice()
        scale = self._masks.satellite_size / self._masks.aperture_size
        target_spectrum = np.nanmean(self._data[target_slice], axis=0)
        target_spectrum *= scale
        target_uncertainty = np.sqrt(
            np.nansum(self._unc[target_slice] ** 2, axis=0)) / n_target
        target_uncertainty *= scale
        data = {
            'target_spectrum': target_spectrum,
            'target_spectrum_unc': target_uncertainty,
        }
        return data

    def _fit_gaussian(self):
        """
        Make a (composite) Gaussian equal to the number of lines in the set and
        fit it. I've added a constant to account for any residual left over
        after background subtraction. I've allowed the Gaussian FWHM to vary
        between the target satellite size (minimum) and slit width (maximum).
        """
        center_indices = [np.abs(self._wavelengths - wavelength).argmin()
                          for wavelength in self._emission_line.wavelengths]
        left_indices = [
            np.abs(self._wavelengths - (wavelength - 0.1*u.nm)).argmin()
            for wavelength in self._emission_line.wavelengths]
        right_indices = [
            np.abs(self._wavelengths - (wavelength + 0.1*u.nm)).argmin()
            for wavelength in self._emission_line.wavelengths]
        sub_indices = []
        for left, right in zip(left_indices, right_indices):
            sub_indices.extend(list(range(left, right+1)))
        sub_indices = np.unique(sub_indices)
        spectrum = self._1d_spectra['target_spectrum']
        spectrum = spectrum.value
        good = np.intersect1d(np.where(~np.isnan(spectrum)), sub_indices)
        x = np.arange(len(spectrum))
        bad = np.array([i for i in x if i not in good])
        n_lines = len(center_indices)
        prefixes = [f'gaussian{i + 1}_' for i in range(n_lines)]
        model = ConstantModel(prefix='constant_')
        model += np.sum([GaussianModel(prefix=prefix) for prefix in prefixes],
                        dtype=object)
        params = Parameters()
        params.add('constant_c', value=0, min=-np.inf, max=np.inf)
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))
        sigma = self._target_width / fwhm_factor
        sigma_min = sigma * 0.75
        sigma_max = self._slit_width / fwhm_factor
        for i, prefix in enumerate(prefixes):
            if i != 0:
                params.add(f'{prefix}amplitude', vary=False,
                           expr=f'gaussian1_amplitude '
                                f'* {self._emission_line.ratios[i]}')
            else:
                params.add(f'{prefix}amplitude', value=np.nanmax(spectrum),
                           min=-np.inf, max=np.inf)
            if i == 0:
                params.add(f'{prefix}center', value=center_indices[i],
                           min=center_indices[i]-self._slit_width/4,
                           max=center_indices[i]+self._slit_width/4)
            else:
                dx = center_indices[i] - center_indices[0]
                params.add(f'{prefix}center', vary=False,
                           expr=f'gaussian1_center + {dx}')
            params.add(f'{prefix}sigma', value=sigma, min=sigma_min,
                       max=sigma_max)
        for method in ['leastsq', 'least_squares', 'nedler']:
            try:
                fit = model.fit(spectrum[good], params=params, x=x[good],
                                method=method)
                if fit.errorbars:
                    break
            except ValueError:
                continue
        return x, bad, fit

    def _get_best_fit(self) -> (u.Quantity, u.Quantity):
        x, bad, fit = self._fit_gaussian()
        constant = fit.params['constant_c'].value
        width = fit.params['gaussian1_sigma'].value
        unit = self._1d_spectra['target_spectrum'].unit
        best_fit = (fit.eval(x=x) - constant) * unit
        best_fit_unc = (fit.eval_uncertainty(x=x)) * unit
        best_fit[bad] = np.nan
        best_fit_unc[bad] = np.nan
        return best_fit, best_fit_unc, width

    def _get_integrated_brightness(self) -> (u.Quantity, u.Quantity):
        dx = np.gradient(self._wavelengths)
        brightness = self._best_fit * dx
        brightness_unc = self._best_fit_unc * dx
        unit = brightness.unit
        brightness = np.nansum(brightness.value) * unit
        brightness_unc = np.sqrt(np.nansum(brightness_unc.value ** 2)) * unit
        return brightness, brightness_unc

    @property
    def target_spectrum(self) -> u.Quantity:
        return self._1d_spectra['target_spectrum']

    @property
    def target_spectrum_unc(self) -> u.Quantity:
        return self._1d_spectra['target_spectrum_unc']

    @property
    def best_fit_spectrum(self) -> u.Quantity:
        return self._best_fit

    @property
    def best_fit_spectrum_unc(self) -> u.Quantity:
        return self._best_fit_unc

    @property
    def best_fit_brightness(self) -> u.Quantity:
        return self._brightness

    @property
    def best_fit_brightness_unc(self) -> u.Quantity:
        return self._brightness_unc

    @property
    def best_fit_width(self) -> float:
        return self._width
