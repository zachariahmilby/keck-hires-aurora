from datetime import datetime
from pathlib import Path

import astropy.units as u

from hiresaurora.data_processing import calibrate_data
from hiresaurora.general import _log, _make_log
from hiresaurora.tabulation import tabulate_results


class AuroraPipeline:

    def __init__(self,
                 reduced_data_directory: str or Path,
                 extended: bool = False,
                 exclude_from_averaging: [int] = None,
                 skip: [str] = None,
                 systematic_trace_offset: float = 0.0,
                 doppler_shift_background: bool = True):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            File path to data reduced with the HIRES data reduction pipeline.
        extended : bool
            Whether or not to use the extended Io lines.
        exclude_from_averaging : [int]
            Indices of observations to exclude from averaging.
        skip : [str]
            Lines to skip when averaging. Example: `skip=['[O I] 557.7 nm']`.
            Default is None.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._extended = extended
        self._exclude = exclude_from_averaging
        self._skip = skip
        self._systematic_trace_offset = systematic_trace_offset
        self._doppler_shift_background = doppler_shift_background
        self._calibrated_data_directory = \
            self._parse_calibrated_data_directory()

    def _parse_calibrated_data_directory(self) -> Path:
        """
        Get the file path to the calibrated data directory.
        """
        return Path(self._reduced_data_directory.parent, 'calibrated')

    def summarize(self) -> None:
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
            excluded=self._exclude, extended=self._extended)

    def run(self,
            trim_bottom: int,
            trim_top: int,
            aperture_radius: u.Quantity) -> None:
        """
        Run the aurora pipeline.

        Parameters
        ----------
        trim_bottom : int
            Number of additional rows to trim off of the bottom of the
            rectified data (the background fitting significantly improves if
            the sawtooth slit edges are excluded). The default is 2.
        trim_top : int
            Number of additional rows to trim off the top of the rectified
            data (the background fitting significantly improves if the sawtooth
            slit edges are excluded). The default is 2.
        aperture_radius : u.Quantity
            The extraction aperture radius in arcsec.

        Returns
        -------
        None.
        """
        t0 = datetime.now()
        log_path = self._parse_calibrated_data_directory()
        _make_log(log_path)
        dataset = self._reduced_data_directory.parent.name
        _log(log_path,
             f"\n{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')}")
        _log(log_path, f'Running aurora calibration pipeline for {dataset}...')
        calibrate_data(
            reduced_data_directory=self._reduced_data_directory,
            extended=self._extended, trim_bottom=trim_bottom,
            trim_top=trim_top, aperture_radius=aperture_radius,
            exclude=self._exclude,
            skip=self._skip,
            systematic_trace_offset=self._systematic_trace_offset,
            doppler_shift_background=self._doppler_shift_background)
        self.summarize()
        elapsed_time = datetime.now() - t0
        _log(log_path, f'Calibration complete, time elapsed: {elapsed_time}.')

    @property
    def reduced_data_directory(self) -> Path:
        return self._reduced_data_directory

    @property
    def calibrated_data_directory(self) -> Path:
        return self._calibrated_data_directory
