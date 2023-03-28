import numpy as np
import statsmodels.api as sm
from scipy.ndimage import median_filter
from astropy.convolution import convolve, Gaussian1DKernel


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
        self._background = self._fit_background()

    @staticmethod
    def _smooth_background(background_profile: np.ndarray) -> np.ndarray:
        return convolve(background_profile, Gaussian1DKernel(stddev=1),
                        boundary='extend')

    def _fit_background(self) -> np.ndarray:
        """
        Fit the normalized background profile to each column to produce a
        characteristic background for a supplied order.
        """
        fit_profile = sm.add_constant(self._slit_profile)
        n_spa, n_spe = self._data.shape
        background = np.zeros((n_spa, n_spe))
        filtered_data = median_filter(self._data, size=(3, 3))
        for column in range(n_spe):
            data = filtered_data[:, column] * self._mask[:, column]
            weights = 1 / self._uncertainty[:, column] ** 2
            try:
                result = sm.WLS(data, fit_profile, weights=weights,
                                missing='drop').fit()
                best_fit_constant = result.params[0]
                best_fit_profile = result.params[1] * self._slit_profile
                background[:, column] = best_fit_constant + best_fit_profile
            except np.linalg.LinAlgError:
                continue

        return background

    @property
    def background(self) -> np.ndarray:
        return self._background
