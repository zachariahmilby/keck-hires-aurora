import warnings

import astropy.units as u
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from lmfit.model import Parameters
from lmfit.models import Model


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
                 mask: np.ndarray, radius: float, spatial_scale: float,
                 spectral_scale: float, smoothed: bool = True):
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
        self._unit = data.unit
        self._mask = mask
        self._radius = radius
        self._spatial_scale = spatial_scale
        self._spectral_scale = spectral_scale
        self._smoothed = smoothed
        self._nspa, self._nspe = self._data.shape
        self._background = np.zeros(self._data.shape)
        self._background_unc = np.zeros(self._data.shape)
        self._fit_row_background()

    def _get_row_profile(self, data: np.ndarray, width=1) -> np.ndarray:
        """
        Create a characteristic normalized row profile over all rows
        without any NaNs.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ind = np.where(~np.isnan(np.sum(data, axis=1)))[0]
            profile = np.nanmean(data[ind], axis=0)
            if self._smoothed:
                kernel = Gaussian1DKernel(stddev=width)
                profile = convolve(profile, kernel, boundary='extend')
            return profile / np.nanmax(profile)

    @staticmethod
    def _fitting_model(profile, coefficient):
        """
        Function to fit a profile to data.
        """
        return coefficient * profile

    def _fit_row_background(self):
        background = np.zeros((self._nspa, self._nspe))
        background_uncertainty = np.zeros((self._nspa, self._nspe))
        masked_data = (self._data - self._background) * self._mask
        uncertainty = np.sqrt(self._uncertainty ** 2
                              + self._background_unc ** 2)
        model = Model(self._fitting_model,
                      independent_vars=['profile'],
                      nan_policy='omit')
        profile = self._get_row_profile(masked_data)
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
                        params.add('coefficient', value=np.nanmax(data), min=0)
                        fit = model.fit(data[good], params=params,
                                        weights=weights[good],
                                        profile=profile[good])
                        with warnings.catch_warnings():
                            warnings.simplefilter(
                                'ignore', category=RuntimeWarning)
                            background[s_] = fit.eval(profile=profile)
                            background_uncertainty[s_] = fit.eval_uncertainty(
                                profile=profile)
                except (ValueError, TypeError):
                    raise BackgroundFitError()
            except BackgroundFitError:
                continue
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
