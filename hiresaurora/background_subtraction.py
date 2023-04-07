import numpy as np
from lmfit.models import Model, LinearModel
from scipy.ndimage import median_filter
from astropy.convolution import convolve, Gaussian1DKernel


class BackgroundFitError(Exception):
    pass


class _Background:
    """
    Fit a background profile to each column.
    """
    def __init__(self, data: np.ndarray, uncertainty: np.ndarray,
                 mask: np.ndarray, slit_profile: np.ndarray):
        self._data = data
        self._uncertainty = uncertainty
        self._mask = mask
        self._slit_profile = self._smooth_background(slit_profile)
        self._primary_background = self._fit_primary_background()
        self._secondary_background = self._fit_secondary_background()

    @staticmethod
    def _smooth_background(background_profile: np.ndarray) -> np.ndarray:
        return convolve(background_profile, Gaussian1DKernel(stddev=1),
                        boundary='extend')

    @staticmethod
    def _primary_fitting_model(profile, constant, coefficient):
        return constant + coefficient * profile

    def _fit_primary_background(self) -> np.ndarray:
        """
        Fit the normalized background profile to each column to produce a
        characteristic background for a supplied order.
        """
        n_spa, n_spe = self._data.shape
        background = np.zeros((n_spa, n_spe))
        filtered_data = median_filter(self._data, size=(3, 3)) * self._mask
        for column in range(n_spe):
            data = filtered_data[:, column]
            weights = 1 / self._uncertainty[:, column] ** 2
            try:
                good = np.where(~np.isnan(data))
                model = Model(self._primary_fitting_model,
                              independent_vars=['profile'],
                              an_policy='omit')
                params = model.make_params(constant=np.nanmin(data),
                                           coefficient=np.nanmean(data))
                fit = model.fit(data[good], params=params,
                                weights=weights[good],
                                profile=self._slit_profile[good])
                background[:, column] = fit.eval(profile=self._slit_profile)
            except ValueError:
                raise BackgroundFitError()

        return background

    # noinspection DuplicatedCode
    def _fit_secondary_background(self):
        """
        Fit a linear background along each row and columnto remove any
        lingering systematic effects remaining from rectification.
        """
        secondary_background_horizontal = np.zeros_like(
            self._primary_background)
        secondary_background_vertical = np.zeros_like(self._primary_background)
        bgsub_data = self._data - self._primary_background
        model = LinearModel()
        x = np.arange(bgsub_data.shape[1])
        for row in range(bgsub_data.shape[0]):
            data = bgsub_data[row] * self._mask[row]
            try:
                good = np.where(~np.isnan(data))
                params = model.guess(data[good], x=x[good])
                fit = model.fit(data[good], params, x=x[good])
                secondary_background_horizontal[row] = fit.eval(x=x)
            except TypeError:
                raise BackgroundFitError()
        y = np.arange(bgsub_data.shape[0])
        for column in range(bgsub_data.shape[1]):
            data = bgsub_data[:, column] * self._mask[:, column]
            try:
                good = np.where(~np.isnan(data))
                params = model.guess(data[good], x=y[good])
                fit = model.fit(data[good], params, x=y[good])
                secondary_background_vertical[:, column] = fit.eval(x=y)
            except TypeError:
                raise BackgroundFitError()
        return secondary_background_horizontal + secondary_background_vertical

    @property
    def background(self) -> np.ndarray:
        return self._primary_background + self._secondary_background
