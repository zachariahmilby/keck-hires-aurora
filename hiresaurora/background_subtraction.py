import warnings

import astropy.units as u
import numpy as np
from astropy.convolution import convolve, interpolate_replace_nans, \
    Gaussian1DKernel, Gaussian2DKernel
from astropy.utils.exceptions import AstropyUserWarning
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
                 spectral_scale: float, interpolate_aperture: bool = False,
                 fill_holes: bool = False):
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
        interpolate_aperture : bool
            If the aperture location appears to be a hole or hill, you can
            replace it with an interpolation based on the surrounding column
            values.
        fill_holes : bool
            If there are bad "holes" left over from background subtraction,
            this will attempt to remove them.
        """
        self._data = data.value
        self._uncertainty = uncertainty.value
        self._unit = data.unit
        self._mask = mask
        self._radius = radius
        self._spatial_scale = spatial_scale
        self._spectral_scale = spectral_scale
        self._nspa, self._nspe = self._data.shape
        self._background = np.zeros(self._data.shape)
        self._background_unc = np.zeros(self._data.shape)
        self._fit_profile_background(kind='column')
        self._smooth_background()
        self._fit_whatevers_left(width=2*self._radius)
        self._fit_profile_background(kind='row')
        self._fit_profile_background(kind='column')
        if interpolate_aperture:
            self._interpolate_aperture()
        if fill_holes:
            for i in range(3):
                self._fill_holes()

    @staticmethod
    def _get_column_profile(data: np.ndarray, width=3) -> np.ndarray:
        """
        Create a characteristic column profile over all columns without any
        NaNs.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ind = np.where(~np.isnan(np.sum(data, axis=0)))[0]
            profile = np.nanmean(data[:, ind], axis=1)
            kernel = Gaussian1DKernel(stddev=width)
            smoothed_profile = convolve(profile, kernel, boundary='extend')
        return smoothed_profile - np.nanmin(smoothed_profile)

    @staticmethod
    def _get_row_profile(data: np.ndarray) -> np.ndarray:
        """
        Create a characteristic row profile, interpolating over the columns
        containing the aperture.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            profile = np.mean(data, axis=0)
            missing_columns = np.where(np.isnan(profile))[0]
            kernel = Gaussian1DKernel(stddev=missing_columns.size)
            profile = interpolate_replace_nans(profile, kernel)
        return profile

    @staticmethod
    def _primary_fitting_model(profile, constant):
        """
        Function to fit a profile to data.
        """
        return constant + profile

    def _apply_smoothing(self, data: np.ndarray, unc: np.ndarray,
                         width: float | int = 2) -> (np.ndarray, np.ndarray):
        """
        Function to convolve a 2-bin-wide Gaussian smoothing kernel to the
        fitted background and uncertainty.
        """
        ratio = self._spectral_scale / self._spatial_scale
        kernel = Gaussian2DKernel(x_stddev=width, y_stddev=width * ratio)
        data = convolve(data, kernel, boundary='extend')
        unc = convolve(np.sqrt(unc ** 2), kernel, boundary='extend')
        return data, unc

    def _fit_profile_background(self, kind: str):
        """
        Fit either a characteristic row or column profile.
        """
        background = np.zeros((self._nspa, self._nspe))
        background_uncertainty = np.zeros((self._nspa, self._nspe))
        masked_data = (self._data - self._background) * self._mask
        uncertainty = np.sqrt(self._uncertainty ** 2
                              + self._background_unc ** 2)
        model = Model(self._primary_fitting_model,
                      independent_vars=['profile'],
                      nan_policy='omit')
        if kind == 'column':
            profile = self._get_column_profile(masked_data)
            n = self._nspe
        elif kind == 'row':
            profile = self._get_row_profile(masked_data)
            n = self._nspa
        else:
            raise Exception
        for i in range(n):
            if kind == 'column':
                s_ = np.s_[:, i]
            elif kind == 'row':
                s_ = np.s_[i]
            else:
                raise Exception
            data = masked_data[s_]
            weights = 1 / uncertainty[s_] ** 2
            try:
                good = np.where(~np.isnan(data))[0]
                if len(good) > 0:
                    params = Parameters()
                    params.add('constant', value=np.nanmin(data))
                    fit = model.fit(data[good], params=params,
                                    weights=weights[good],
                                    profile=profile[good])
                    background[s_] = fit.eval(profile=profile)
                    background_uncertainty[s_] = fit.eval_uncertainty(
                        profile=profile)
            except ValueError:
                raise BackgroundFitError()
        self._background += background
        self._background_unc = np.sqrt(self._background_unc**2
                                       + background_uncertainty**2)

    def _fit_whatevers_left(self, width):
        """
        Do one final pass at eliminating any residual structure. Based on what
        I think happens with Sextractor.
        """
        masked_data = (self._data - self._background) * self._mask
        uncertainty = np.sqrt(self._uncertainty ** 2
                              + self._background_unc ** 2)
        cycle = True
        while cycle:
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=AstropyUserWarning)
                try:
                    masked_data, uncertainty = self._apply_smoothing(
                        data=masked_data, unc=uncertainty, width=width)
                except AstropyUserWarning:
                    width += 0.5
                else:
                    cycle = False
        self._background += masked_data
        self._background_unc = np.sqrt(self._background_unc ** 2
                                       + uncertainty ** 2)

    def _smooth_background(self):
        """
        Smooth the current form of the background and uncertainty with a
        Gaussian kernel of a given width.
        """
        self._background, self._background_unc = self._apply_smoothing(
            data=self._background, unc=self._background_unc)

    def _interpolate_aperture(self):
        """
        Make sure the aperture isn't a hill or hole.
        """
        masked_data = (self._data - self._background) * self._mask
        row_profile = np.mean(masked_data, axis=0)
        missing_columns = np.where(np.isnan(row_profile))[0]
        for col in missing_columns:
            ind = np.where(np.isnan(masked_data[:, col]))[0]
            bg_slice = self._background[:, col]
            bg_slice[ind] = np.nan
            kernel = Gaussian1DKernel(stddev=ind.size/2)
            bg_slice = interpolate_replace_nans(bg_slice, kernel,
                                                boundary='extend')
            self._background[:, col] = bg_slice

    def _fill_holes(self):
        """
        Use Astropy NaN interpolation to fill residual holes from fitting.
        """
        data = self._data - self._background
        masked_data = data * self._mask
        std = np.nanstd(masked_data)
        ind = np.where(data < -std)
        data[ind] = np.nan
        ratio = self._spectral_scale / self._spatial_scale
        kernel = Gaussian2DKernel(x_stddev=1, y_stddev=1*ratio)
        data = interpolate_replace_nans(data, kernel, boundary='extend')
        self._data = data + self._background

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
