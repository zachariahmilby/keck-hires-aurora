import warnings

import astropy.units as u
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from lmfit.model import Parameters
from lmfit.models import Model, RectangleModel, PolynomialModel, ConstantModel

from hiresaurora.general import known_emission_lines
from hiresaurora.masking import _Mask


class BackgroundFitError(Exception):
    """
    Exception for failing to fit background.
    """
    pass


class _Background:
    """
    Fit a background to masked data.
    """

    def __init__(self, data: u.Quantity, uncertainty: u.Quantity,
                 rest_wavelengths: u.Quantity, slit_width_bins: float,
                 mask: _Mask, radius: float, spatial_scale: float,
                 spectral_scale: float, fit_residual: bool = False):
        """
        Parameters
        ----------
        data : u.Quantity
            The raw data.
        uncertainty : u.Quantity
            The raw data uncertainty.
        mask : np.ndarray
            The target mask (an array of ones and NaNs).
        spatial_scale : float
            The spatial pixel scale (probably in [arcsec/bin], but it doesn't
            matter so long as it's the same units as the spectral scale).
        spectral_scale : float
            The spectral pixel scale (probably in [arcsec/bin], but it doesn't
            matter so long as it's the same units as the spatial scale).
        """
        self._data = data.value
        self._uncertainty = uncertainty.value
        self._rest_wavelengths = rest_wavelengths
        self._slit_width_bins = slit_width_bins
        self._unit = data.unit
        self._mask = mask
        self._radius = radius
        self._spatial_scale = spatial_scale
        self._spectral_scale = spectral_scale
        self._fit_residual = fit_residual
        self._nspa, self._nspe = self._data.shape
        self._background = np.zeros(self._data.shape)
        self._background_unc = np.zeros(self._data.shape)
        self._fit_background()

    def _fit_background(self):
        self._fit_row_background(degree=5)
        if self._fit_residual:
            self._fit_row_background(degree=1)

    def _get_profile(self):
        smoothed_data = (self._data -
                         self._background).copy() * self._mask.target_mask
        kernel = Gaussian1DKernel(stddev=4)
        for col in range(smoothed_data.shape[1]):
            smoothed_data[:, col] = convolve(
                smoothed_data[:, col], kernel, boundary='extend')
        return np.mean(smoothed_data, axis=0)

    def _get_skyline(self) -> np.ndarray:
        """
        Fit a known Earth sky emission line.
        """
        profile = self._get_profile()
        sky_line = np.zeros_like(profile)
        for wavelength in known_emission_lines:
            ind = np.abs(self._rest_wavelengths - wavelength).argmin()
            halfwidth = int(self._slit_width_bins / 2)
            if (ind != 0) & (ind != self._rest_wavelengths.size - 1):
                s_ = np.s_[ind-halfwidth-10:ind+halfwidth+10]
                x = np.arange(profile.size)
                model = RectangleModel(form='logistic') + ConstantModel()
                params = Parameters()
                params.add('c', value=0)
                params.add(
                    'amplitude', value=np.nanmax(profile), min=0)
                params.add('width', value=self._slit_width_bins,
                           min=self._slit_width_bins * 0.9,
                           max=self._slit_width_bins * 1.1)
                params.add('center', value=ind)
                params.add('center1', expr='center-width/2')
                params.add('center2', expr='center+width/2')
                params.add('sigma1', value=0.85, min=0.7, max=1)
                params.add('sigma2', expr='sigma1')
                fit = model.fit(profile[s_], params, x=x[s_])
                sky_line = fit.eval_components(x=x)['rectangle']
        return sky_line

    def _get_characteristic_background(self) -> np.ndarray:
        """
        Create a characteristic normalized row profile over all rows
        without any NaNs.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            profile = self._get_profile()
            sky_line = self._get_skyline()
            profile -= sky_line
            kernel = Gaussian1DKernel(stddev=2)
            profile = convolve(profile, kernel, boundary='extend')
            return profile

    def _construct_fitting_profile(self) -> np.ndarray:
        sky_line = self._get_skyline()
        background = self._get_characteristic_background()
        profile = background + sky_line
        return profile / np.nanmax(profile)

    @staticmethod
    def _fitting_func(profile, coefficient):
        """
        Function to fit a profile to data.
        """
        return coefficient * profile

    def _fitting_model(self):
        model = Model(self._fitting_func,
                      independent_vars=['profile'],
                      nan_policy='omit')
        return model

    def _fit_row_background(self, degree: int = 5):
        background = np.zeros((self._nspa, self._nspe))
        background_uncertainty = np.zeros((self._nspa, self._nspe))
        masked_data = (self._data - self._background) * self._mask.target_mask
        uncertainty = np.sqrt(self._uncertainty ** 2
                              + self._background_unc ** 2)
        coefficients = np.zeros(self._nspa)
        profile = self._construct_fitting_profile()
        model = self._fitting_model()
        for i in range(self._nspa):
            try:
                s_ = np.s_[i]
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        'ignore', category=RuntimeWarning)
                    data = masked_data[s_]
                    weights = 1 / uncertainty[s_] ** 2
                try:
                    good = np.where(~np.isnan(data))[0]
                    if len(good) > 0:
                        params = Parameters()
                        params.add('coefficient', value=np.nanmax(data))
                        fit = model.fit(data[good], params=params,
                                        weights=weights[good],
                                        profile=profile[good])
                        coefficients[i] = fit.params['coefficient'].value
                except (ValueError, TypeError):
                    raise BackgroundFitError()
            except BackgroundFitError:
                continue
        model = PolynomialModel(degree=degree)
        x = np.arange(coefficients.size)
        s_ = np.s_[1:-1]  # ignore order edges when fitting
        params = model.guess(coefficients[s_], x=x[s_])
        fit = model.fit(coefficients[s_], params, x=x[s_])
        coefficient_fit = fit.eval(x=x)
        coefficient_fit_unc = fit.eval_uncertainty(params, x=x)
        for i in range(coefficients.size):
            background[i] = profile * coefficient_fit[i]
            background_uncertainty[i] = profile * coefficient_fit_unc[i]
        self._background += background
        self._background_unc = np.sqrt(self._background_unc**2
                                       + background_uncertainty**2)

    @property
    def best_fit(self) -> u.Quantity:
        return self._background * self._unit

    @property
    def best_fit_uncertainty(self) -> u.Quantity:
        return self._background_unc * self._unit

    @property
    def data(self) -> u.Quantity:
        return (self._data - self._background) * self._unit

    @property
    def uncertainty(self) -> u.Quantity:
        return np.sqrt(self._uncertainty**2
                       + self._background_unc**2) * self._unit
