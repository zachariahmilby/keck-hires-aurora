from datetime import datetime
from pathlib import Path

import astropy.units as u

from hiresaurora.data_processing import calibrate_data
from hiresaurora.tabulation import tabulate_results


class AuroraPipeline:

    def __init__(self, reduced_data_directory: str or Path,
                 extended: bool = False, exclude_from_averaging: [int] = None):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            File path to data reduced with the HIRES data reduction pipeline.
        extended : bool
            Whether or not to use the extended Io lines.
        exclude_from_averaging : [int]
            Indices of observations to exclude from averaging.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._extended = extended
        self._exclude = exclude_from_averaging
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

    def run(self, trim_bottom: int, trim_top: int, aperture_radius: u.Quantity,
            average_aperture_scale: float, horizontal_offset: int or float,
            average_trace_offset: int or float = 0.0,
            interpolate_aperture: bool = False,
            fill_holes: bool = False) -> None:
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
        average_aperture_scale : float
            Factor to scale the aperture radius by for the average images.
        horizontal_offset : int or float
            Any additional offset if the wavelength solution is off.
        average_trace_offset : int or float
            Additional vertical offset for "trace" in the average image.
        interpolate_aperture : bool
            If the aperture location appears to be a hole or hill, you can
            replace it with an interpolation based on the surrounding column
            values.
        fill_holes : bool
            If there are bad "holes" left over from background subtraction,
            this will attempt to remove them.

        Returns
        -------
        None.
        """
        t0 = datetime.now()
        dataset = self._reduced_data_directory.parent.name
        print(f'Running aurora pipeline for {dataset}...')
        calibrate_data(reduced_data_directory=self._reduced_data_directory,
                       extended=self._extended, trim_bottom=trim_bottom,
                       trim_top=trim_top, aperture_radius=aperture_radius,
                       average_aperture_scale=average_aperture_scale,
                       horizontal_offset=horizontal_offset,
                       exclude=self._exclude,
                       average_trace_offset=average_trace_offset,
                       interpolate_aperture=interpolate_aperture,
                       fill_holes=fill_holes)
        self.summarize()
        elapsed_time = datetime.now() - t0
        print(f'Processing complete, time elapsed: {elapsed_time}.')

    @property
    def reduced_data_directory(self) -> Path:
        return self._reduced_data_directory

    @property
    def calibrated_data_directory(self) -> Path:
        return self._calibrated_data_directory
