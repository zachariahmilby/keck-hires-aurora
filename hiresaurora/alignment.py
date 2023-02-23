import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from lmfit.models import GaussianModel


class _TraceOffsets:
    """
    Calculate the fractional vertical position in each rectified frame where
    the trace falls.
    """
    def __init__(self, directory: str or Path, order: int, trim_bottom: int,
                 trim_top: int):
        self._directory = Path(directory)
        self._order = order
        self._trim_bottom = trim_bottom
        self._trim_top = trim_top
        self._centers = self._find_centers()

    @staticmethod
    def _choose_sample_points(n_pixels: int):
        """
        Determine the positions to sample along the trace.
        """
        return np.arange(0, n_pixels, 256, dtype=int)

    def _find_centers(self) -> np.ndarray:
        """
        Calculate the average center position in each order for each
        observation. The resulting array has the shape (n_files, n_orders,).
        """
        files = \
            sorted(Path(self._directory, 'guide_satellite').glob('*.fits.gz'))
        model = GaussianModel()
        file_centers = []
        for i, file in enumerate(files):
            with fits.open(file) as hdul:
                data = hdul['PRIMARY'].data[
                       self._order, self._trim_bottom:-self._trim_top]
            ind = self._choose_sample_points(data.shape[1])
            x = np.arange(data.shape[0])
            centers = np.full(len(ind), fill_value=np.nan)
            unc = np.full(len(ind), fill_value=np.nan)
            for j, pos in enumerate(ind):
                y = data[:, pos]
                if len(np.where(np.isnan(y))[0]) != 0:
                    continue
                params = model.guess(y, x=x)
                fit = model.fit(y, params, x=x)
                centers[j] = fit.params['center'].value
                unc[j] = fit.params['center'].stderr
            average_centers = np.nansum(unc * centers) / np.nansum(unc)
            file_centers.append(average_centers)
        return np.array(file_centers)

    def align_and_average(
            self, data: np.ndarray,
            unc: np.ndarray) -> ([np.ndarray], [np.ndarray]):
        """
        Align target science data based on the vertical positions then average
        all observations together order-by-order. Propagate uncertainty
        accordingly. Because the offsets vary order-to-order, the returned
        objects are lists instead of arrays.
        """
        integer_centers = np.round(self._centers).astype(int)
        n_obs, n_spa, n_spe = data.shape

        # get centers for this order and offset them from their minimum
        centers = integer_centers
        centers -= np.min(centers)

        # make an empty array to hold the aligned data
        aligned_data = np.full(
            (n_obs, n_spa + np.max(centers), n_spe), fill_value=np.nan)
        aligned_unc = np.full(
            (n_obs, n_spa + np.max(centers), n_spe), fill_value=np.nan)

        # loop through each observation and place it in the proper position
        for obs in range(n_obs):
            aligned_data[obs, centers[obs]:centers[obs]+n_spa] = data[obs]
            aligned_unc[obs, centers[obs]:centers[obs]+n_spa] = unc[obs]

        # average data and propagate uncertainty
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            average_order_data = np.nanmean(aligned_data, axis=0)
            average_order_unc = np.sqrt(
                np.nansum(aligned_unc ** 2, axis=0)) / n_obs

        return average_order_data[:n_spa], average_order_unc[:n_spa]

    @property
    def centers(self) -> np.ndarray:
        return self._centers
