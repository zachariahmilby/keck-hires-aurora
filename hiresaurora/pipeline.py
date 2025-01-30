from datetime import datetime
from pathlib import Path

import astropy.units as u
import pytz

from hiresaurora.data_processing import calibrate_data
from hiresaurora.general import _log, _make_log
from hiresaurora.tabulation import tabulate_results


class AuroraPipeline:

    def __init__(self,
                 reduced_data_directory: str or Path,
                 fit_background: bool = True,
                 extended: bool = False,
                 exclude_from_averaging: list[int] = None,
                 skip: list[str] = None,
                 systematic_trace_offset: dict or int or float = 0.0,
                 horizontal_offset: float = 0.0,
                 doppler_shift_background: bool = True,
                 smooth: list[str] = None):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            File path to data reduced with the HIRES data reduction pipeline.
        fit_background : bool
            Whether or not to calculate a best-fit background. Default is True.
        extended : bool
            Whether or not to use the extended Io lines.
        exclude_from_averaging : [int]
            Indices of observations to exclude from averaging.
        skip : [str]
            Lines to skip when calculating brightnesses.
            Example: `skip=['557.7 nm [O I]']`. Default is None.
        systematic_trace_offset : dict or int or float
            Additional vertical offset for the "trace" in all images. To apply
            to all orders/wavelengths, pass an int or float. To apply just to
            a single line, pass a dictionary where the key(s) are the specific
            lines to which you want to apply the offset. For the dictionary
            value you can pass an int or float or a list of ints or
            floats. If you pass a list, it should have the same length as the
            number of individual observations. For example, if there are 3
            observations and you want to systematically offset the second trace
            downward by 3 bins for the 777.4 nm O I line, you would pass
            `systematic_trace_offset={'777.4 nm O I': [0, -2, 0]}`.
        horizontal_offset : float
                A manual horizontal offset in spectral bins. Useful if the
                wavelength solution is off or the Doppler shift is wrong.
        doppler_shift_background : bool
            Whether or not to allow the background to Doppler shift. If the slit
            wasn't very long or the signal isn't very strong, this might produce
            bad results and should be turned off.
        smooth : list[str]
            List of specific lines for which the template background spectrum
            should be smoothed before fitting.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._fit_background = fit_background
        self._extended = extended
        self._exclude = exclude_from_averaging
        self._skip = skip
        self._systematic_trace_offset = systematic_trace_offset
        self._horizontal_offset = horizontal_offset
        self._doppler_shift_background = doppler_shift_background
        self._smooth = smooth
        self._calibrated_data_directory = \
            self._parse_calibrated_data_directory()

    def _parse_calibrated_data_directory(self) -> Path:
        """
        Get the file path to the calibrated data directory.
        """
        return Path(self._reduced_data_directory.parent, 'calibrated')

    def _summarize(self) -> None:
        """
        Generate a summary table and graphic.

        Parameters
        ----------
        None.

        Returns
        -------
        None.
        """
        tabulate_results(
            calibrated_data_path=self._calibrated_data_directory,
            excluded=self._exclude,
            extended=self._extended)

    def run(self,
            aperture_radius: u.Quantity,
            trim_bottom: int = 2,
            trim_top: int = 2,
            average_aperture_scale: float = 1.0) -> None:
        """
        Run the aurora pipeline.

        Parameters
        ----------
        aperture_radius : u.Quantity
            The extraction aperture radius in arcsec.
        trim_bottom : int
            Number of additional rows to trim off of the bottom of the
            rectified data (the background fitting significantly improves if
            the sawtooth slit edges are excluded). The default is 2.
        trim_top : int
            Number of additional rows to trim off the top of the rectified
            data (the background fitting significantly improves if the sawtooth
            slit edges are excluded). The default is 2.
        average_aperture_scale : float
            Scaling factor for the average aperture in case it needs to be
            larger than the individual apertures. Default is 1.0.

        Returns
        -------
        None.
        """
        t0 = datetime.now()
        log_path = self._parse_calibrated_data_directory()
        _make_log(log_path)
        dataset = self._reduced_data_directory.parent.name
        _log(log_path,
             f"\n{datetime.now(tz=pytz.utc).strftime('%Y-%m-%d %H:%M:%S.%f')}")
        _log(log_path, f'Running aurora calibration pipeline for {dataset}...')
        calibrate_data(
            reduced_data_directory=self._reduced_data_directory,
            fit_background=self._fit_background,
            extended=self._extended,
            trim_bottom=trim_bottom,
            trim_top=trim_top,
            aperture_radius=aperture_radius,
            average_aperture_scale=average_aperture_scale,
            exclude=self._exclude,
            skip=self._skip,
            systematic_trace_offset=self._systematic_trace_offset,
            horizontal_offset=self._horizontal_offset,
            doppler_shift_background=self._doppler_shift_background,
            smooth=self._smooth)
        self._summarize()
        elapsed_time = datetime.now() - t0
        _log(log_path, f'Calibration complete, time elapsed: {elapsed_time}.')

    @property
    def reduced_data_directory(self) -> Path:
        """
        Absolute path to reduced data directory.
        """
        return self._reduced_data_directory.absolute()

    @property
    def calibrated_data_directory(self) -> Path:
        """
        Absolute path to calibrated data directory.
        """
        return self._calibrated_data_directory.absolute()
