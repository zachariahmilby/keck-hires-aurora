import warnings

import numpy as np
import statsmodels.api as sm


class _Background:
    """
    Fit a background profile to each column.
    """
    def __init__(self, data: np.ndarray, uncertainty: np.ndarray,
                 mask: np.ndarray, slit_profile: np.ndarray):
        self._data = data
        self._uncertainty = uncertainty
        self._mask = mask
        self._slit_profile = slit_profile
        self._background = self._fit_background()

    def _fit_background(self) -> np.ndarray:
        """
        Fit the normalized background profile to each column to produce a
        characteristic background for a supplied order.
        """
        fit_profile = sm.add_constant(self._slit_profile)
        n_spa, n_spe = self._data.shape
        background = np.zeros((n_spa, n_spe))
        for column in range(n_spe):
            result = sm.WLS(
                self._data[:, column] * self._mask[:, column], fit_profile,
                weights=1/self._uncertainty[:, column]**2,
                missing='drop').fit()
            best_fit_constant = result.params[0]
            best_fit_profile = result.params[1] * self._slit_profile
            background[:, column] = best_fit_constant + best_fit_profile
        return background

    @property
    def background(self) -> np.ndarray:
        return self._background
