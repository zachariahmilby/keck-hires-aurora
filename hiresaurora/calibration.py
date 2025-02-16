import warnings
from pathlib import Path

import astropy.constants as c
import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time

from hiresaurora.ephemeris import _get_ephemeris
from hiresaurora.general import _package_directory, _doppler_shift_wavelengths


def _get_solar_spectral_brightness() -> tuple[u.Quantity, u.Quantity]:
    """
    This function retrieves the theoretical solar spectral radiance above
    Earth's atmosphere at 1 au from the Sun from 320 to 1000 nm at 1 nm
    resolution. The data are actually spectral irradiance (W/m²/nm) which I
    convert to radiance by dividing by pi, giving units of W/m²/nm/sr. I
    then convert to photon brightness by dividing by photon energy for each
    wavelength. The spectral irradiance data come from the
    Coddington et al. (2023) TSIS-1 Hybrid Solar Reference Spectrum (HSRS).
    """
    filepath = Path(_package_directory, 'anc',
                    'solar_spectral_irradiance.dat')
    wavelength, irradiance = np.genfromtxt(filepath, delimiter=',',
                                           skip_header=True, unpack=True)
    wavelength = wavelength * u.nm
    irradiance = irradiance * u.W / u.m ** 2 / u.nm
    radiance = irradiance / (np.pi * u.sr)
    photon_energy = c.c * c.h / wavelength / u.photon  # noqa
    brightness = radiance / photon_energy
    return wavelength.to(u.nm), brightness.to(u.R / u.nm)


class _FluxCalibration:
    """
    Calcualtes flux calibrations and their corresponding uncertainty from
    Jupiter meridian observations. Results are in (electrons/s/sr)/(R/nm).
    """
    def __init__(self,
                 reduced_data_directory: str or Path,
                 wavelengths: u.Quantity,
                 order: int,
                 trim_bottom: int,
                 trim_top: int):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            Path to the reduced data directory.
        wavelengths : u.Quantity
            Wavelength(s) to use for calibration. For multiplet lines the
            average is used.
        order : int
            The order containing the spectrum with the targeted wavelength(s).
        trim_bottom : int
            Number of rows to trim from the top of the 2D spectrum.
        trim_top : int
            Number of rows to trim from the bottom of the 2D spectrum.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._wavelength = wavelengths.mean().to(u.nm).value
        self._order = order
        self._trim_bottom = trim_bottom
        self._trim_top = trim_top
        self._calibration_factors = self._calculate_calibration_factor()

    def _get_flux_calibration_file(self):
        """
        Get a list of paths for the Jupiter flux calibration files.
        """
        filepath = Path(self._reduced_data_directory, 'flux_calibration')
        return sorted(filepath.glob('*.fits.gz'))[0]

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _get_jupiter_meridian_reflectivity() -> tuple[u.Quantity, u.Quantity]:
        """
        This function retrieves reflectivity (also called I/F) for Jupiter's
        meridian from 320 to 1000 nm at 1 nm resolution. I stole these data
        from figures 1 and 6 in Woodman et al. (1979) "Spatially Resolved
        Reflectivities of Jupiter during the 1976 Opposition"
        (doi: 10.1016/0019-1035(79)90116-7).
        """
        filepath = Path(_package_directory, 'anc',
                        'jupiter_meridian_reflectivity.dat')
        wavelength, reflectivity = np.genfromtxt(filepath, delimiter=',',
                                                 skip_header=True, unpack=True)
        wavelength = wavelength * u.nm
        reflectivity = reflectivity * u.dimensionless_unscaled
        return wavelength, reflectivity

    def _get_jupiter_spectral_brightness(
            self,
            time: Time) -> tuple[u.Quantity, u.Quantity]:
        """
        Load the solar reference spectrum and Jupiter's meridian reflectivity
        and multiply them.
        """
        wavelength, reflectivity = self._get_jupiter_meridian_reflectivity()
        _, brightness = _get_solar_spectral_brightness()
        jupiter_brightness = reflectivity * brightness
        eph = _get_ephemeris(target='Jupiter', time=time, location='@10')
        distance = eph['delta'].value[0] * eph['delta'].unit
        scaling_factor = (u.au / distance.to(u.au)) ** 2
        scaled_brightness = jupiter_brightness * scaling_factor
        return wavelength, scaled_brightness

    # noinspection PyUnresolvedReferences
    def _calculate_calibration_factor(self) -> dict[str, u.Quantity]:
        """
        Calculate the calibration factor and uncertainty for a specific
        wavelength in units of (electrons/s/sr)/(R/nm).
        """

        # make slice for order and trims
        select = np.s_[self._order, self._trim_bottom:-self._trim_top]

        # load the first calibration file to get ancillary information
        flux_calibration_file = self._get_flux_calibration_file()
        with fits.open(flux_calibration_file) as hdul:
            header = hdul['PRIMARY'].header
            date = header['DATE-OBS']
            data = hdul['PRIMARY'].data[select]
            uncertainty = hdul['PRIMARY_UNC'].data[select]
            slit_area = header['SLITWID'] * header['SLITLEN'] * u.arcsec ** 2
            n_bins = header['SLITWIDB'] * header['SLITLENB']
            slit_width_bins = header['SLITWIDB']
            wavelengths = \
                hdul['BIN_CENTER_WAVELENGTHS'].data[self._order] * u.nm

        # remove doppler shift from wavelengths
        time = Time(header['DATE-OBS'], format='isot', scale='utc')
        eph = _get_ephemeris(target='Jupiter',
                             time=time)
        time -= eph['lighttime'].quantity[0]
        eph = _get_ephemeris(target='Jupiter',
                             time=time)
        velocity = eph['delta_rate'].quantity[0]
        shifted_wavelengths = _doppler_shift_wavelengths(wavelengths, velocity)

        # get spectral location for wavelength
        spec_ind = np.abs(
            shifted_wavelengths.value - self._wavelength).argmin()

        # load Jupiter's spectral brightness
        spectral_wavelength, spectral_brightness = \
            self._get_jupiter_spectral_brightness(time=time)

        # add units to each array
        data *= u.electron / u.s
        uncertainty *= u.electron / u.s

        # set output unit
        output_unit = (u.electron / u.s / u.sr) / (u.R / u.nm)

        # average across spatial dimension
        average_flux = np.nanmean(data[:, spec_ind])
        average_unc = np.sqrt(
            np.nansum(uncertainty[:, spec_ind]**2)) / uncertainty.shape[0]

        # propagate average to whole slit, then divide by slit area
        electron_flux = average_flux * n_bins / slit_area
        electron_flux_unc = average_unc * n_bins / slit_area

        # spreads out over whole slit, so divide by slit width in bins
        electron_flux /= slit_width_bins
        electron_flux_unc /= slit_width_bins

        # interpolate Jupiter spectral brightness over wavelengths in order
        photon_flux = spectral_brightness[
            np.abs(spectral_wavelength.value - self._wavelength).argmin()]

        # convert factors to output units
        calibration_factor = (electron_flux / photon_flux).to(output_unit)
        calibration_factor_unc = (electron_flux_unc /
                                  photon_flux).to(output_unit)

        # return the factors
        return {'calibration_factor': calibration_factor,
                'calibration_factor_unc': calibration_factor_unc}

    def calibrate(self,
                  data: u.Quantity,
                  unc: u.Quantity,
                  target_size: u.Quantity) -> tuple[u.Quantity, u.Quantity]:
        """
        Wrapper function to calibrate data and uncertainty arrays from
        electrons/s to rayleighs/nm. Calibration assumes emission from a disk
        with the angular size of the target satellite.

        Parameters
        ----------
        data : u.Quantity
            The data array.
        unc : u.Quantity
            The uncertainty array.
        target_size : u.Quantity
            The target solid angular size.

        Returns
        -------
        The calibrated data and ucnertainty.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            data /= target_size
            unc /= target_size
            calibrated_data = (
                    data / self._calibration_factors['calibration_factor'])
            data_nsr = unc / data
            calibration_nsr = (
                    self._calibration_factors['calibration_factor_unc'] /
                    self._calibration_factors['calibration_factor'])
            calibrated_unc = np.abs(calibrated_data) * np.sqrt(
                data_nsr ** 2 + calibration_nsr ** 2)
        return calibrated_data.to(u.R / u.nm), calibrated_unc.to(u.R / u.nm)

    @property
    def calibration_factor(self) -> u.Quantity:
        """
        Calibration factor for specified wavelength in units of
        (electrons/s/sr)/(R/nm).
        """
        return self._calibration_factors['calibration_factor']

    @property
    def calibration_factors_unc(self) -> u.Quantity:
        """
        Calibration factor uncertainty for a specified wavelength in units of
        (electrons/s/sr)/(R/nm).
        """
        return self._calibration_factors['calibration_factor_unc']
