import warnings

import astropy.units as u
import numpy as np
from astropy.convolution import convolve
from lmfit.model import ModelResult
from lmfit.models import Model, RectangleModel, PolynomialModel
from lmfit.parameter import Parameters

from hiresaurora.general import (known_emission_lines,
                                 _doppler_shift_wavelengths, _slit_kernel)
from hiresaurora.masking import _Mask

warnings.simplefilter('ignore', category=RuntimeWarning)


class BackgroundFitError(Exception):
    """General error for background fitting issues."""
    pass


class _Background:
    """
    Fit scattered light background and Earth atmospheric sky lines.
    """
    def __init__(self,
                 data_2d: u.Quantity,
                 uncertainty_2d: u.Quantity,
                 rest_wavelengths: u.Quantity,
                 shifted_wavelength_centers: u.Quantity,
                 shifted_wavelength_edges: u.Quantity,
                 slit_width_bins: float,
                 mask: _Mask,
                 radius: float,
                 spatial_scale: float,
                 spectral_scale: float,
                 allow_doppler_shift: bool,
                 smooth: bool = False,
                 fit_skyline: bool = True):
        """
        Parameters
        ----------
        data_2d : u.Quantity
            The raw 2D data.
        uncertainty_2d : u.Quantity
            The raw 2D data uncertainty.
        mask : np.ndarray
            The target mask (an array of ones and NaNs).
        spatial_scale : float
            The spatial pixel scale (probably in [arcsec/bin], but it doesn't
            matter so long as it's the same units as the spectral scale).
        spectral_scale : float
            The spectral pixel scale (probably in [arcsec/bin], but it doesn't
            matter so long as it's the same units as the spatial scale).
        """
        self._data_2d = data_2d.value
        self._uncertainty_2d = uncertainty_2d.value
        self._rest_wavelengths = rest_wavelengths
        self._shifted_wavelength_centers = shifted_wavelength_centers
        self._shifted_wavelength_edges = shifted_wavelength_edges
        self._slit_width_bins = slit_width_bins
        self._unit = data_2d.unit
        self._mask = mask
        self._radius = radius
        self._spatial_scale = spatial_scale
        self._spectral_scale = spectral_scale
        self._allow_doppler_shift = allow_doppler_shift
        self._smooth = smooth
        self._nspa, self._nspe = self._data_2d.shape
        self._background_2d = np.zeros(self._data_2d.shape)
        self._background_unc_2d = np.zeros(self._data_2d.shape)
        self._background_1d = np.zeros(self._data_2d.shape[1])
        self._background_unc_1d = np.zeros(self._data_2d.shape[1])
        self._sky_line = np.zeros(self._data_2d.shape)
        self._profile = self._get_background_profile()
        if fit_skyline:
            self._remove_skyline()
        self._fit_2d_background()

    @staticmethod
    def _background_fit_function(background: np.ndarray,
                                 coeff: float,
                                 const: float) -> np.ndarray:
        """
        Function to build a template fitting model with Lmfit.
        """
        return background * coeff + const

    def _background_fit_model(self) -> Model:
        """
        Lmfit background template fitting model.
        """
        model = Model(self._background_fit_function,
                      independent_vars=['background'],
                      nan_policy='omit')
        model.set_param_hint('coeff', min=0)
        return model

    @staticmethod
    def _doppler_shift_background(background: np.ndarray,
                                  wavelengths: u.Quantity,
                                  velocity: float) -> np.ndarray:
        """
        Doppler shift background using linear interpolation.
        """
        shifted_wavelengths = _doppler_shift_wavelengths(
            wavelengths, velocity * u.km / u.s)
        shifted_background = np.interp(
            wavelengths, shifted_wavelengths, background)
        return shifted_background

    def _doppler_fit_function(self,
                              background: np.ndarray,
                              wavelengths: u.Quantity,
                              velocity: float,
                              coeff: float,
                              const: float) -> np.ndarray:
        """
        Function to build a template fitting model with Lmfit.
        """
        shifted_background = self._doppler_shift_background(
            background, wavelengths, velocity)
        return shifted_background * coeff + const

    def _doppler_fit_model(self, velocity_limit: float = 10) -> Model:
        """
        Lmfit background template fitting model.
        """
        model = Model(self._doppler_fit_function,
                      independent_vars=['background', 'wavelengths'])
        if self._allow_doppler_shift:
            model.set_param_hint('velocity', min=-velocity_limit,
                                 max=velocity_limit)
        else:
            model.set_param_hint('velocity', value=0.0, vary=False)
        return model

    def _find_doppler_shift(self,
                            data: np.ndarray,
                            wavelengths: u.Quantity,
                            background: np.ndarray) -> ModelResult:
        """
        Determine optimal Doppler shift of background relative to a template.
        Used for aligning individual rows when constructing a master template.
        """
        model = self._doppler_fit_model()
        params = model.make_params(
            coeff=np.max(data), velocity=0.0, const=0.0)
        fit = model.fit(data, params, background=background,
                        wavelengths=wavelengths)
        return fit

    def _get_doppler_shift_along_slit(self, profile: np.ndarray) -> np.ndarray:
        masked_data = (self._data_2d - self._sky_line -
                       self._background_2d) * self._mask.target_mask
        velocities = np.zeros(self._nspa)
        if not self._allow_doppler_shift:
            return velocities
        velocities_unc = np.zeros(self._nspa)
        for i in range(self._nspa):
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore', category=RuntimeWarning)
                data = masked_data[i]
                good = np.where(~np.isnan(data))[0]
                fit = self._find_doppler_shift(
                    data[good], wavelengths=self._rest_wavelengths[good],
                    background=profile[good])
                velocities[i] = fit.params['velocity'].value
                velocities_unc[i] = fit.params['velocity'].stderr
        good = ~np.isnan(velocities) & ~np.isnan(velocities_unc)
        model = PolynomialModel(degree=2)
        x = np.arange(self._nspa)
        params = model.guess(velocities[good], x=x[good])
        fit = model.fit(velocities[good], params,
                        weights=1/velocities_unc[good]**2, x=x[good])
        return fit.eval(x=x)

    # noinspection DuplicatedCode
    def _get_background_profile(self) -> np.ndarray or None:
        """
        Construct a normalized, characteristic background spectrum for fitting
        individual rows.
        """
        data = (self._data_2d - self._sky_line -
                self._background_2d) * self._mask.target_mask

        keep1 = []
        for col in range(data.shape[1]):
            if ~np.isnan(data[:, col]).all():
                keep1.append(col)
        keep1 = np.array(keep1)

        slit_profile = np.mean(data[:, keep1], axis=1)
        rows = np.where(~np.isnan(slit_profile))[0]
        if len(rows) == 0:
            return None
        aligned_rows = np.full_like(data, fill_value=np.nan)
        use_row = 0
        for row in range(self._nspa):
            if np.nansum(data[row]) != 0:
                use_row = row
                break
        other_rows = np.array([i for i in rows if i != use_row])
        aligned_rows[use_row] = data[use_row]
        for row in other_rows:
            good = np.where(~np.isnan(data[use_row]))[0]
            fit = self._find_doppler_shift(
                data[use_row][good], wavelengths=self._rest_wavelengths[good],
                background=data[row][good])
            aligned_rows[row] = fit.eval(wavelengths=self._rest_wavelengths,
                                         background=data[row])
        median_background = np.nanmedian(aligned_rows, axis=0)

        sky_line = np.zeros_like(median_background)
        for wavelength in known_emission_lines:
            ind = np.abs(self._rest_wavelengths - wavelength).argmin()
            halfwidth = int(self._slit_width_bins / 2)
            if (ind > 0) & (ind < self._nspe - 1):
                s_ = np.s_[ind-halfwidth-10:ind+halfwidth+10]
                x = np.arange(median_background.size)
                model = (RectangleModel(form='logistic') +
                         PolynomialModel(degree=1))
                params = Parameters()
                params.add('amplitude', value=np.nanmax(median_background),
                           min=0)
                params.add('width', value=self._slit_width_bins-2*0.6)
                params.add('center', value=ind, min=ind-5, max=ind+5)
                params.add('center1', expr='center-width/2')
                params.add('center2', expr='center+width/2')
                params.add('sigma1', value=0.6)
                params.add('sigma2', value=0.6)
                params.add('c0', value=0.0)
                params.add('c1', value=0.0)
                with warnings.catch_warnings():
                    warnings.simplefilter('error', category=RuntimeWarning)
                    try:
                        fit = model.fit(median_background[s_], params,
                                        x=x[s_])  # noqa
                        try:
                            amplitude = fit.params['amplitude'].value
                            amplitude_err = fit.params['amplitude'].stderr
                            amplitude_snr = amplitude / amplitude_err
                            center1 = fit.params['center1'].value
                            center1_err = fit.params['center1'].stderr
                            center1_snr = center1 / center1_err
                            center2 = fit.params['center2'].value
                            center2_err = fit.params['center2'].stderr
                            center2_snr = center2 / center2_err
                            sigma1 = fit.params['sigma1'].value
                            sigma1_err = fit.params['sigma1'].stderr
                            sigma1_snr = sigma1 / sigma1_err
                            sigma2 = fit.params['sigma2'].value
                            sigma2_err = fit.params['sigma2'].stderr
                            sigma2_snr = sigma2 / sigma2_err
                            if ((amplitude_snr < 1) or (center1_snr < 1) or
                                (center2_snr < 1) or (sigma1_snr < 1) or
                                (sigma2_snr < 1)):
                                continue
                            else:
                                sky_line = fit.eval_components(x=x)['rectangle']
                        except TypeError:
                            continue
                    except RuntimeWarning:
                        continue
        median_background -= sky_line

        # smooth background if fewer than 3 rows were used to make it or
        # explicitly called for
        if (len(other_rows) < 3) or self._smooth:
            smoothed_background = convolve(
                median_background, _slit_kernel(self._slit_width_bins),
                boundary='extend')
            normalized_background = (smoothed_background /
                                     np.nanmax(smoothed_background))
        else:
            normalized_background = (median_background /
                                     np.nanmax(median_background))
        return normalized_background

    @staticmethod
    def _make_sky_line_image(data: np.ndarray,
                             dxdy: float,
                             width: float,
                             center: float,
                             sigma: float) -> np.ndarray:
        """
        Construct a 2D image of a sky line using defined slit shape parameters.
        """
        image = np.zeros_like(data)
        x = np.arange(image.shape[1])
        model = RectangleModel(form='logistic')
        for row in range(image.shape[0]):
            params = Parameters()
            params.add('amplitude', value=1, vary=False)
            params.add('width', value=width)
            params.add('center', value=center + dxdy * row)
            params.add('center1', expr='center-width/2')
            params.add('center2', expr='center+width/2')
            params.add('sigma1', value=sigma)
            params.add('sigma2', value=sigma)
            image[row] = model.eval(params, x=x)
        return image

    def _sky_line_fit_func(self,
                           data_arr: np.ndarray,
                           ind: np.ndarray,
                           coeff: float,
                           dxdy: float,
                           width: float,
                           center: float,
                           sigma: float) -> np.ndarray:
        """
        Function to build a sky line fitting model with Lmfit.
        """
        sky_line_image = self._make_sky_line_image(
            data=data_arr, dxdy=dxdy, width=width, center=center, sigma=sigma)
        sky_line_image = sky_line_image.flatten()[ind.astype(int)]
        return sky_line_image * coeff

    def _fit_sky_line_2d(self, background) -> None:
        """
        Lmfit sky line template fitting model.
        """
        model = Model(
            self._sky_line_fit_func, independent_vars=['data_arr', 'ind'])
        for wavelength in known_emission_lines:
            ind = np.abs(self._rest_wavelengths - wavelength).argmin()
            if (ind > 0) & (ind < self._nspe - 1):
                masked_data = \
                    (self._data_2d - background) * self._mask.target_mask
                model.set_param_hint('coeff', min=0)
                model.set_param_hint(
                    'dxdy', min=-self._slit_width_bins/10,
                    max=self._slit_width_bins/10)
                model.set_param_hint(
                    'width', min=self._slit_width_bins*0.5,
                    max=self._slit_width_bins*1.5)
                model.set_param_hint(
                    'center', min=ind-self._slit_width_bins*2,
                    max=ind+self._slit_width_bins*2)
                model.set_param_hint('sigma', min=0.5, max=1)
                params = model.make_params(
                    coeff=np.nanmean(masked_data[:, ind]), dxdy=0.0,
                    width=self._slit_width_bins, center=float(ind), sigma=0.6)
                flattened_data = masked_data.flatten()
                ind = np.where(~np.isnan(flattened_data))[0]
                fit = model.fit(
                    flattened_data[ind], params, data_arr=masked_data, ind=ind)
                x = np.arange(flattened_data.size)
                best_fit_image = fit.eval(
                    data_arr=masked_data, ind=x).reshape(masked_data.shape)
                self._sky_line = best_fit_image

    def _fit_row_background(self,
                            mask_skyline: bool) -> None:
        """
        Fit background row by row. If fit_skyline is True, remove the sky line
        from the background template. If mask_skyline is True, mask out the
        sky line when fitting the background.
        """
        if self._profile is None:
            raise BackgroundFitError

        background = np.zeros((self._nspa, self._nspe))
        background_uncertainty = np.zeros((self._nspa, self._nspe))
        masked_data = (self._data_2d - self._sky_line -
                       self._background_2d) * self._mask.target_mask
        uncertainty = np.sqrt(self._uncertainty_2d ** 2
                              + self._background_unc_2d ** 2)
        model = self._background_fit_model()

        # ensure Doppler velocity varies smoothly along the slit
        velocities = self._get_doppler_shift_along_slit(profile=self._profile)

        if mask_skyline:
            for wavelength in known_emission_lines:
                ind = np.abs(self._rest_wavelengths - wavelength).argmin()
                halfwidth = int(2.5 * self._slit_width_bins / 2)
                if (ind > 0) & (ind < self._nspe - 1):
                    sky_line_mask = np.ones_like(masked_data)
                    sky_line_mask[:, ind-halfwidth:ind+halfwidth+1] = np.nan
                    masked_data *= sky_line_mask
        for i in range(self._nspa):
            with warnings.catch_warnings():
                warnings.simplefilter(
                    'ignore', category=RuntimeWarning)
                data = masked_data[i]
                weights = 1 / uncertainty[i] ** 2
                params = model.make_params(
                    coeff=np.nanmean(data), const=0.0)
                shifted_profile = self._doppler_shift_background(
                    self._profile, self._rest_wavelengths, velocities[i])  # noqa
                good = np.where(~np.isnan(data) & ~np.isnan(weights) &
                                ~np.isnan(shifted_profile))[0]
                if len(good) > 0:
                    fit = model.fit(
                        data[good], params=params,
                        background=shifted_profile[good])
                    background[i] += fit.eval(
                        background=shifted_profile)
                    background_uncertainty[i] += fit.eval_uncertainty(
                        background=shifted_profile)

        for col in range(background.shape[1]):
            if np.sum(background[:, col]) == 0:
                background[:, col] = np.nan

        # mask skyline fits and removes known skylines from 2D data
        if mask_skyline:
            self._fit_sky_line_2d(background)
        else:
            self._background_2d += background
            self._background_unc_2d = np.sqrt(
                self._background_unc_2d ** 2 + background_uncertainty ** 2)

    def _remove_skyline(self) -> None:
        self._fit_row_background(mask_skyline=True)

    def _fit_2d_background(self) -> None:
        self._fit_row_background(mask_skyline=False)

    @property
    def best_fit_2d(self) -> u.Quantity:
        return self._background_2d * self._unit

    @property
    def best_fit_uncertainty_2d(self) -> u.Quantity:
        return self._background_unc_2d * self._unit

    @property
    def data_2d(self) -> u.Quantity:
        return self._data_2d * self._unit

    @property
    def uncertainty_2d(self) -> u.Quantity:
        return self._uncertainty_2d * self._unit

    @property
    def sky_line_fit(self) -> u.Quantity:
        return self._sky_line * self._unit

    @property
    def bg_sub_data_2d(self) -> u.Quantity:
        return (self._data_2d - self._sky_line -
                self._background_2d) * self._unit

    @property
    def bg_sub_uncertainty_2d(self) -> u.Quantity:
        return np.sqrt(self._uncertainty_2d**2
                       + self._background_unc_2d**2) * self._unit
