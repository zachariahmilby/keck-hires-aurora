import warnings
from copy import deepcopy
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from hirespipeline.files import make_directory
from lmfit.model import ModelResult
from lmfit.models import PolynomialModel, GaussianModel, ConstantModel
from scipy.ndimage import shift

from hiresaurora.background_subtraction import _Background, BackgroundFitError
from hiresaurora.calibration import _FluxCalibration
from hiresaurora.ephemeris import _get_ephemeris
from hiresaurora.general import _doppler_shift_wavelengths, AuroraLines, _log
from hiresaurora.graphics import make_quicklook
from hiresaurora.masking import _Mask
from hiresaurora.observing_geometry import Geometry


class TraceFitError(Exception):
    pass


def _fit_trace(data: u.Quantity or np.ndarray,
               unc: u.Quantity or np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate fractional pixel vertical position of the trace.
    """
    gaussian_model = GaussianModel()
    constant_model = ConstantModel()
    composite_model = gaussian_model + constant_model
    x = np.arange(data.shape[0])
    centers = np.full(data.shape[1], fill_value=np.nan)
    centers_unc = np.full_like(centers, fill_value=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        for i in range(data.shape[1]):
            column = data[:, i]
            weights = 1 / unc[:, i] ** 2
            if isinstance(data, u.Quantity):
                column = column.value
                weights = weights.value
            good = ~np.isnan(column)
            if len(good) < data.shape[0] - 3:  # skip if more than 3 NaNs
                continue
            try:
                gaussian_params = gaussian_model.guess(column[good], x=x[good])
                constant_params = constant_model.guess(column[good], x=x[good])
                composite_params = gaussian_params + constant_params
                fit = composite_model.fit(
                    column[good], composite_params, weights=weights[good],
                    x=x[good], nan_policy='omit')
                centers[i] = fit.params['center'].value
                centers_unc[i] = fit.params['center'].stderr
            except (ValueError, IndexError, TypeError):
                raise TraceFitError()
        try:
            good = ~np.isnan(centers) & ~np.isnan(centers_unc)
            model = PolynomialModel(degree=0)
            x = np.arange(data.shape[1])
            params = model.guess(centers[good], x=x[good])
            fit = model.fit(centers[good], params,
                            weights=1/centers_unc[good] ** 2, x=x[good])
        except (ValueError, IndexError, TypeError):
            raise TraceFitError()
        return fit.params['c0'].value, fit.params['c0'].stderr


class WavelengthNotFoundError(Exception):
    """
    Raised if a wavelength isn't found in available solutions.
    """
    pass


class _Wavelengths:
    """
    Class to hold wavelengths in rest and Doppler-shifted frames.
    """
    def __init__(self, wavelengths: u.Quantity, relative_velocity: u.Quantity):
        self._wavelengths = wavelengths
        self._relative_velocity = relative_velocity
        self._doppler_shifted_wavelengths = _doppler_shift_wavelengths(
            self._wavelengths, self._relative_velocity)

    @property
    def rest(self) -> u.Quantity:
        return self._wavelengths

    @property
    def doppler_shifted(self) -> u.Quantity:
        return self._doppler_shifted_wavelengths


class _ImageData:
    """
    Class to hold FITS data and header information after conversion to Astropy
    quantity.
    """
    def __init__(self, hdul: fits.HDUList):
        self._filename = Path(hdul.filename()).name
        self._data = self._convert_data_to_quantity(hdul['PRIMARY'])
        self._data_header = hdul['PRIMARY'].header
        self._unc = self._convert_data_to_quantity(hdul['PRIMARY_UNC'])
        self._unc_header = hdul['PRIMARY_UNC'].header
        self._ephemeris = self._get_target_ephemeris()
        self._wavelength_centers = self._get_wavelengths(
            hdul['BIN_CENTER_WAVELENGTHS'])
        self._wavelength_edges = self._get_wavelengths(
            hdul['BIN_EDGE_WAVELENGTHS'])
        self._echelle_orders = hdul['ECHELLE_ORDERS'].data

    @staticmethod
    def _convert_data_to_quantity(hdu: fits.FitsHDU) -> u.Quantity:
        """
        Parse header unit and apply to data.
        """
        return hdu.data * u.Unit(hdu.header['BUNIT'])

    def _get_ephemeris_quantity(self, item: str) -> u.Quantity:
        """
        Retrieve an item from the target ephemeris.
        """
        return self._ephemeris[item].value[0] * self._ephemeris[item].unit

    def _get_target_ephemeris(self) -> dict:
        """
        Query the JPL Horizons ephemeris tool to get ephemeris table.
        """
        target = self._data_header['TARGET']
        time = Time(self._data_header['DATE-OBS'], format='isot', scale='utc')
        ephemeris = _get_ephemeris(
            target=target, time=time, skip_daylight=False,
            airmass_lessthan=None)
        return ephemeris

    def _get_wavelengths(self, hdu: fits.FitsHDU) -> _Wavelengths:
        """
        Get the Doppler-shifted wavelengths.
        """
        data = self._convert_data_to_quantity(hdu)
        wavelengths = _Wavelengths(
            wavelengths=data,
            relative_velocity=self._get_ephemeris_quantity('delta_rate'))
        return wavelengths

    @property
    def filename(self) -> str:
        return self._filename

    @property
    def data(self) -> u.Quantity:
        return self._data

    @property
    def data_header(self) -> fits.Header:
        return self._data_header

    @property
    def uncertainty(self) -> u.Quantity:
        return self._unc

    @property
    def uncertainty_header(self) -> fits.Header:
        return self._unc_header

    @property
    def rest_wavelength_centers(self) -> u.Quantity:
        return self._wavelength_centers.rest

    @property
    def doppler_shifted_wavelength_centers(self) -> u.Quantity:
        return self._wavelength_centers.doppler_shifted

    @property
    def rest_wavelength_edges(self) -> u.Quantity:
        return self._wavelength_edges.rest

    @property
    def doppler_shifted_wavelength_edges(self) -> u.Quantity:
        return self._wavelength_edges.doppler_shifted

    @property
    def angular_radius(self) -> u.Quantity:
        return self._get_ephemeris_quantity('ang_width') / 2

    @property
    def relative_velocity(self) -> u.Quantity:
        return self._get_ephemeris_quantity('delta_rate')


class _RawData:
    """
    Class to hold the raw science and trace data.
    """
    def __init__(self, reduced_data_directory: str or Path):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            File path to the reduced data directory from the HIRES data
            reduction pipeline.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._science_data = self._get_data('science')
        self._trace_data = self._get_data('guide_satellite')

    def _get_data(self, directory: str) -> [_ImageData]:
        """
        Retrieve raw data from FITS files in a given directory.
        """
        hduls = []
        files = sorted(
            Path(self._reduced_data_directory, directory).glob('*.fits.gz'))
        for file in files:
            with fits.open(file) as hdul:
                hduls.append(_ImageData(hdul))
        return hduls

    def find_order_with_wavelength(self, wavelength: u.Quantity) -> int:
        """
        Find which order index contains a user-supplied wavelength.
        """
        average_wavelength = wavelength.to(u.nm).mean()
        found_order = None
        for data in self._science_data:
            wavelengths = data.doppler_shifted_wavelength_centers
            for order in range(wavelengths.shape[0]):
                ind = np.abs(wavelengths[order] - average_wavelength).argmin()
                if (ind > 0) & (ind < wavelengths.shape[1] - 1):
                    found_order = order
                    break
        if found_order is None:
            raise WavelengthNotFoundError(
                f'Sorry, {average_wavelength} not found!')
        return found_order

    @property
    def science(self) -> list[_ImageData]:
        return self._science_data

    @property
    def trace(self) -> list[_ImageData]:
        return self._trace_data


# noinspection DuplicatedCode
class _LineData:
    """
    Process individual data for a given emisison line or multiplet.
    """
    def __init__(self,
                 reduced_data_directory: str or Path,
                 aperture_radius: u.Quantity,
                 trim_top: int = 2,
                 trim_bottom: int = 2,
                 systematic_trace_offset: float = 0.0,
                 doppler_shift_background: bool = False):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            File path to the reduced data directory from the HIRES data
            reduction pipeline.
        aperture_radius : u.Quantity
            The extraction aperture radius in arcsec.
        trim_top : int
            Number of additional rows to trim off the top of the rectified
            data (the background fitting significantly improves if the sawtooth
            slit edges are excluded). The default is 2.
        trim_bottom : int
            Number of additional rows to trim off of the bottom of the
            rectified data (the background fitting significantly improves if
            the sawtooth slit edges are excluded). The default is 2.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._aperture_radius = aperture_radius.to(u.arcsec)
        self._trim_top = trim_top
        self._trim_bottom = trim_bottom
        self._systematic_trace_offset = systematic_trace_offset
        self._doppler_shift_background = doppler_shift_background
        self._data = _RawData(self._reduced_data_directory)
        self._save_directory = self._parse_save_directory()
        self._sigma = None

    def _parse_save_directory(self) -> Path:
        return Path(self._reduced_data_directory.parent, 'calibrated')

    def _get_data_slice(self,
                        order: int,
                        line_wavelengths: u.Quantity,
                        dwavelength: u.Quantity = 0.5*u.nm) -> (np.s_, np.s_):
        """
        Get slice corresponding to ±dwavelength from Doppler-shifted
        wavelength. Uses minimum and maximum of entire observing sequence to
        account for variation in the bounds due to changing Doppler shift.
        Makes slices for both data and wavelength edges (one additional entry
        along horizontal axis).
        """

        keep1 = []
        masked_data = self._data.science[0].data[order]
        masked_data[np.where(masked_data == 0)] = np.nan
        for col in range(masked_data.shape[1]):
            if ~np.isnan(masked_data[:, col]).all():
                keep1.append(col)
        keep_left = np.min(keep1) + 2
        keep_right = np.max(keep1) - 2

        lefts = []
        rights = []
        left_bound = np.min(line_wavelengths) - dwavelength
        right_bound = np.max(line_wavelengths) + dwavelength
        for data in self._data.science:
            wavelengths = data.doppler_shifted_wavelength_centers[order]
            left = np.abs(wavelengths - left_bound).argmin()
            if left < 0:
                left = 0
            right = np.abs(wavelengths - right_bound).argmin() + 1
            if right > wavelengths.shape[0]:
                right = wavelengths.shape[0]
            lefts.append(left)
            rights.append(right)
        left = np.max([np.min(lefts), keep_left])
        right = np.min([np.max(rights), keep_right])
        data_slice = np.s_[self._trim_bottom:-self._trim_top, left:right]
        edge_slice = np.s_[self._trim_bottom:-self._trim_top, left:right+1]
        return data_slice, edge_slice

    @staticmethod
    def _get_line_indices(wavelengths: u.Quantity,
                          line_wavelengths: u.Quantity) -> [float]:
        """
        Get fractional pixel indices corresponding to emission line locations.
        """
        model = PolynomialModel(degree=3)
        pixels = np.arange(wavelengths.shape[0])
        params = model.guess(pixels, x=wavelengths.value)
        fit = model.fit(pixels, params, x=wavelengths.value)
        indices = []
        for wavelength in line_wavelengths:
            indices.append(fit.eval(x=wavelength.value))
        return indices

    @staticmethod
    def _fix_individual_header(data: fits.ImageHDU, unit: u.Unit or None,
                               comment: str = None):
        """
        Fix entries in an individual observation header.
        """
        data.header['NAXIS1'] = (data.header['NAXIS1'],
                                 'number of spectral bins')
        if 'NAXIS2' in data.header:
            data.header['NAXIS2'] = (data.header['NAXIS2'],
                                     'number of spatial bins')
        if unit is not None:
            data.header.set('BUNIT', f'{unit}', 'data physical units')
        if comment is not None:
            data.header['COMMENT'] = comment
        for key in list(data.header.keys()):
            if key not in ['XTENSION', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1',
                           'NAXIS2', 'GCOUNT', 'PCOUNT', 'BUNIT', 'AIRMASS',
                           'DATE-OBS', 'TARGET', 'EXPTIME', 'IMAGETYP',
                           'EXTNAME']:
                del data.header[key]

    @staticmethod
    def _fix_average_header(data: fits.ImageHDU, unit: u.Unit or None,
                            comment: str = None):
        """
        Fix entries in an average observation header.
        """
        data.header['NAXIS1'] = (data.header['NAXIS1'],
                                 'number of spectral bins')
        if 'NAXIS2' in data.header:
            data.header['NAXIS2'] = (data.header['NAXIS2'],
                                     'number of spatial bins')
        if unit is not None:
            data.header.set('BUNIT', f'{unit}', 'data physical units')
        if comment is not None:
            data.header['COMMENT'] = comment
        for key in list(data.header.keys()):
            if key not in ['XTENSION', 'SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1',
                           'NAXIS2', 'GCOUNT', 'PCOUNT', 'BUNIT', 'TARGET',
                           'IMAGETYP', 'EXTNAME']:
                del data.header[key]

    def _make_image_hdu(self, data: u.Quantity or np.ndarray,
                        header: fits.Header or None,
                        name: str, comment: str = None,
                        average: bool = False) -> fits.ImageHDU:
        """
        Construct an ImageHDU (non-PrimaryHDU) and give it an appropriate
        header.
        """
        if isinstance(data, u.Quantity):
            hdu = fits.ImageHDU(data=data.value, header=header, name=name)
            if not average:
                self._fix_individual_header(hdu, unit=data.unit,
                                            comment=comment)
            else:
                self._fix_average_header(hdu, unit=data.unit, comment=comment)
        else:
            hdu = fits.ImageHDU(data=data, header=header, name=name)
            if not average:
                self._fix_individual_header(hdu, unit=None, comment=comment)
            else:
                self._fix_average_header(hdu, unit=None, comment=comment)
        return hdu

    @staticmethod
    def _round(number: float, precision: int = 4) -> float:
        """
        My attempt at curbing rounding errors (like lots of 9s...).
        """
        return float(np.format_float_positional(number, precision=precision))

    def _set_primary_header(self,
                            primary_hdu: fits.PrimaryHDU,
                            unit: u.Unit,
                            line_name: str,
                            brightness: u.Quantity, brightness_unc: u.Quantity,
                            trace: float = None, trace_unc: float = None,
                            t_corr: Time = None,
                            lat_mag: u.Quantity = None,
                            lon_mag: u.Quantity = None,
                            orb_dist: u.Quantity = None,
                            height: u.Quantity = None,
                            lat_obs: u.Quantity = None,
                            lon_obs: u.Quantity = None,
                            angular_radius: u.Quantity = None,
                            relative_velocity: u.Quantity = None,
                            distance_to_target: u.Quantity = None,
                            north_pole_angle: u.Quantity = None,
                            ):
        """
        Set primary extension header information (not observation-specific).
        """
        del primary_hdu.header['TARGET']
        del primary_hdu.header['AIRMASS']
        del primary_hdu.header['EXPTIME']
        primary_hdu.header['BUNIT'] = f'{unit}'
        primary_hdu.header.set('LINE', f'{line_name}',
                               'targeted emission line')
        primary_hdu.header.set('BRGHT', brightness.value,
                               f'best-fit brightness [{brightness.unit}]')
        primary_hdu.header.set('BRGHTUNC', brightness_unc.value,
                               f'best-fit brightness [{brightness.unit}]')
        if (trace is not None) and (trace != 'error'):
            if (trace_unc is not None) and (trace != 'error'):
                primary_hdu.header.set('TRACEFIT', trace,
                                       'trace fit fractional pixel location')
                primary_hdu.header.set(
                    'TRACEUNC', trace_unc,
                    'trace fit fractional pixel uncertainty')
        else:
            primary_hdu.header.set('TRACEFIT', trace,
                                   'trace fit fractional pixel location')
            primary_hdu.header.set('TRACEUNC', trace_unc,
                                   'trace fit fractional pixel location')
        if t_corr is not None:
            primary_hdu.header.set('DATECORR', t_corr.isot,
                                   'time corrected for light travel time')
        if lat_mag is not None:
            primary_hdu.header.set(
                'MAGLAT', self._round(lat_mag.value, 4),
                f'target magnetic latitude [{lat_mag.unit}]')
            primary_hdu.header.set(
                'MAGLON', self._round(lon_mag.value, 4),
                f'target magnetic east longitude [{lon_mag.unit}]')
            primary_hdu.header.set(
                'ORB_DIST', self._round(orb_dist.value, 4),
                f'target orbital distance from Jupiter [{orb_dist.unit}]')
            primary_hdu.header.set(
                'PS_DIST', self._round(height.value, 4),
                f'target distance from plamsa sheet [{height.unit}]')
            primary_hdu.header.set(
                "OBSLAT", self._round(lat_obs.value, 4),
                f'sub-observer latitude [{lat_obs.unit}]')
            primary_hdu.header.set(
                "OBSLON", self._round(lon_obs.value, 4),
                f'sub-observer east longitude [{lon_obs.unit}]')
        if angular_radius is not None:
            primary_hdu.header.set(
                'ANG_RAD', self._round(angular_radius.value, 4),
                f'target angular radius [{angular_radius.unit}]')
        if relative_velocity is not None:
            relative_velocity = relative_velocity.to(u.km/u.s)
            primary_hdu.header.set(
                'RELVLCTY', relative_velocity.value,
                f'target relative velocity [{relative_velocity.unit}]')
        if distance_to_target is not None:
            distance_to_target = distance_to_target.to(u.km)
            primary_hdu.header.set(
                'TARGDIST', int(self._round(distance_to_target.value, 0)),
                f'distance to target [{distance_to_target.unit}]')
        if north_pole_angle is not None:
            north_pole_angle = north_pole_angle.to(u.degree)
            primary_hdu.header.insert(
                'SKYPA',
                ('NPANG', self._round(north_pole_angle.value, 4),
                 f'north pole angle [{north_pole_angle.unit}]'),
                after=True)

    # noinspection DuplicatedCode
    def save_individual_fits(
            self,
            line: u.Quantity,
            line_name: str,
            data_header: fits.Header,
            trace_fit: float,
            trace_fit_unc: float,
            geometry: Geometry,
            angular_radius: u.Quantity,
            relative_velocity: u.Quantity,
            target_masks: np.ndarray,
            background_masks: np.ndarray,
            aperture_edges: np.ndarray,
            image_data: u.Quantity,
            image_data_unc: u.Quantity,
            image_background: u.Quantity,
            image_skyline: u.Quantity,
            spectrum_1d: u.Quantity,
            spectrum_1d_unc: u.Quantity,
            best_fit_1d: u.Quantity,
            best_fit_1d_unc: u.Quantity,
            wavelength_centers_rest: u.Quantity,
            wavelength_edges_rest: u.Quantity,
            wavelength_centers_shifted: u.Quantity,
            wavelength_edges_shifted: u.Quantity,
            brightness: u.Quantity,
            brightness_unc: u.Quantity,
            save_directory: Path,
            file_name: str):
        """
        Save calibrated data for an individual observation as a FITS file.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', category=fits.verify.VerifyWarning)

            primary_hdu = fits.PrimaryHDU(data=image_data.value,
                                          header=data_header.copy())
            self._set_primary_header(
                primary_hdu=primary_hdu,
                unit=image_data.unit,
                line_name=line_name,
                trace=trace_fit, trace_unc=trace_fit_unc,
                angular_radius=angular_radius,
                relative_velocity=relative_velocity,
                t_corr=geometry.light_corrected_time,
                lat_mag=geometry.magnetic_latitude,
                lon_mag=geometry.magnetic_longitude,
                orb_dist=geometry.orbital_distance,
                height=geometry.height,
                lat_obs=geometry.sub_observer_latitude,
                lon_obs=geometry.sub_observer_longitude,
                distance_to_target=geometry.distance,
                north_pole_angle=geometry.north_pole_angle,
                brightness=brightness,
                brightness_unc=brightness_unc)
            primary_unc_hdu = self._make_image_hdu(
                data=image_data_unc, header=None, name='PRIMARY_UNC',
                comment='Primary (wavelength-integrated imaging data) '
                        'uncertainty.')
            primary_background_hdu = self._make_image_hdu(
                data=image_background, header=data_header.copy(),
                name='BACKGROUND_FIT',
                comment='Best-fit background.')
            primary_skyline_hdu = self._make_image_hdu(
                data=image_skyline, header=data_header.copy(),
                name='SKYLINE_FIT',
                comment='Best-fit Earth sky line.')
            targeted_lines_hdu = self._make_image_hdu(data=line, header=None,
                                                      name='TARGETED_LINES')
            target_mask_hdu = self._make_image_hdu(
                data=target_masks, header=None, name='TARGET_MASKS',
                comment='Mask(s) used to isolate background when calculating '
                        'background fit.')
            background_mask_hdu = self._make_image_hdu(
                data=background_masks, header=None, name='BACKGROUND_MASKS',
                comment='Mask(s) used to isolate target when calculating '
                        'emission brightness.')
            aperture_edges_hdu = self._make_image_hdu(
                data=aperture_edges, header=None, name='APERTURE_EDGES',
                comment='Pixel coordinates of edges of target aperture.')
            spectrum_1d_hdu = self._make_image_hdu(
                data=spectrum_1d, header=None, name='SPECTRUM_1D')
            spectrum_1d_unc_hdu = self._make_image_hdu(
                data=spectrum_1d_unc, header=None, name='SPECTRUM_1D_UNC')
            best_fit_1d_hdu = self._make_image_hdu(
                data=best_fit_1d, header=None, name='BEST_FIT_1D')
            best_fit_1d_unc_hdu = self._make_image_hdu(
                data=best_fit_1d_unc, header=None, name='BEST_FIT_1D_UNC')
            rest_wavelength_centers_hdu = self._make_image_hdu(
                data=wavelength_centers_rest, header=None,
                name='WAVELENGTH_CENTERS_REST',
                comment='Rest frame (Earth) wavelengths.')
            rest_wavelength_edges_hdu = self._make_image_hdu(
                data=wavelength_edges_rest, header=None,
                name='WAVELENGTH_EDGES_REST',
                comment='Rest frame (Earth) wavelengths.')
            shifted_wavelength_centers_hdu = self._make_image_hdu(
                data=wavelength_centers_shifted, header=None,
                name='WAVELENGTH_CENTERS_SHIFTED',
                comment='Doppler-shifted wavelengths.')
            shifted_wavelength_edges_hdu = self._make_image_hdu(
                data=wavelength_edges_shifted, header=None,
                name='WAVELENGTH_EDGES_SHIFTED',
                comment='Doppler-shifted wavelengths.')

            hdus = [primary_hdu,
                    primary_unc_hdu,
                    primary_background_hdu,
                    primary_skyline_hdu,
                    targeted_lines_hdu,
                    target_mask_hdu,
                    background_mask_hdu,
                    aperture_edges_hdu,
                    spectrum_1d_hdu,
                    spectrum_1d_unc_hdu,
                    best_fit_1d_hdu,
                    best_fit_1d_unc_hdu,
                    rest_wavelength_centers_hdu,
                    rest_wavelength_edges_hdu,
                    shifted_wavelength_centers_hdu,
                    shifted_wavelength_edges_hdu,
                    ]
            hdul = fits.HDUList(hdus)
            output_directory = Path(save_directory, line_name)
            make_directory(output_directory)
            savename = Path(output_directory,
                            file_name.replace('reduced', 'calibrated'))
            hdul.writeto(savename, overwrite=True)
            hdul.close()
            return savename

    # noinspection DuplicatedCode
    def save_average_fits(
            self,
            line: u.Quantity,
            line_name: str,
            data_header: fits.Header,
            tracefit: float,
            angular_radius: u.Quantity,
            target_masks: np.ndarray,
            background_masks: np.ndarray,
            aperture_edges: np.ndarray,
            image_data: u.Quantity,
            image_data_unc: u.Quantity,
            image_background: u.Quantity,
            image_skyline: u.Quantity,
            spectrum_1d: u.Quantity,
            spectrum_1d_unc: u.Quantity,
            best_fit_1d: u.Quantity,
            best_fit_1d_unc: u.Quantity,
            wavelength_centers_rest: u.Quantity,
            wavelength_edges_rest: u.Quantity,
            wavelength_centers_shifted: u.Quantity,
            wavelength_edges_shifted: u.Quantity,
            brightness: u.Quantity,
            brightness_unc: u.Quantity,
            save_directory: Path):
        """
        Save calibrated data for an average observation as a FITS file.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', category=fits.verify.VerifyWarning)

            primary_hdu = fits.PrimaryHDU(data=image_data.value,
                                          header=data_header.copy())
            self._set_primary_header(
                primary_hdu=primary_hdu,
                unit=image_data.unit,
                line_name=line_name,
                trace=tracefit,
                angular_radius=angular_radius,
                brightness=brightness,
                brightness_unc=brightness_unc)
            primary_unc_hdu = self._make_image_hdu(
                data=image_data_unc, header=None, name='PRIMARY_UNC',
                comment='Primary (wavelength-integrated imaging data) '
                        'uncertainty.')
            primary_background_hdu = self._make_image_hdu(
                data=image_background, header=data_header.copy(),
                name='BACKGROUND_FIT',
                comment='Best-fit background.')
            primary_skyline_hdu = self._make_image_hdu(
                data=image_skyline, header=data_header.copy(),
                name='SKYLINE_FIT',
                comment='Best-fit Earth sky line.')
            targeted_lines_hdu = self._make_image_hdu(
                data=line, header=None, name='TARGETED_LINES')
            target_mask_hdu = self._make_image_hdu(
                data=target_masks, header=None, name='TARGET_MASKS',
                comment='Mask(s) used to isolate background when calculating '
                        'background fit.')
            background_mask_hdu = self._make_image_hdu(
                data=background_masks, header=None, name='BACKGROUND_MASKS',
                comment='Mask(s) used to isolate target when calculating '
                        'emission brightness.')
            aperture_edges_hdu = self._make_image_hdu(
                data=aperture_edges, header=None, name='APERTURE_EDGES',
                comment='Pixel coordinates of edges of target aperture.')
            spectrum_1d_hdu = self._make_image_hdu(
                data=spectrum_1d, header=None, name='SPECTRUM_1D')
            spectrum_1d_unc_hdu = self._make_image_hdu(
                data=spectrum_1d_unc, header=None, name='SPECTRUM_1D_UNC')
            best_fit_1d_hdu = self._make_image_hdu(
                data=best_fit_1d, header=None, name='BEST_FIT_1D')
            best_fit_1d_unc_hdu = self._make_image_hdu(
                data=best_fit_1d_unc, header=None, name='BEST_FIT_1D_UNC')
            rest_wavelength_centers_hdu = self._make_image_hdu(
                data=wavelength_centers_rest, header=None,
                name='WAVELENGTH_CENTERS_REST',
                comment='Rest frame (Earth) wavelengths.')
            rest_wavelength_edges_hdu = self._make_image_hdu(
                data=wavelength_edges_rest, header=None,
                name='WAVELENGTH_EDGES_REST',
                comment='Rest frame (Earth) wavelengths.')
            shifted_wavelength_centers_hdu = self._make_image_hdu(
                data=wavelength_centers_shifted, header=None,
                name='WAVELENGTH_CENTERS_SHIFTED',
                comment='Doppler-shifted wavelengths.')
            shifted_wavelength_edges_hdu = self._make_image_hdu(
                data=wavelength_edges_shifted, header=None,
                name='WAVELENGTH_EDGES_SHIFTED',
                comment='Doppler-shifted wavelengths.')
            hdus = [primary_hdu,
                    primary_unc_hdu,
                    primary_background_hdu,
                    primary_skyline_hdu,
                    targeted_lines_hdu,
                    target_mask_hdu,
                    background_mask_hdu,
                    aperture_edges_hdu,
                    spectrum_1d_hdu,
                    spectrum_1d_unc_hdu,
                    best_fit_1d_hdu,
                    best_fit_1d_unc_hdu,
                    rest_wavelength_centers_hdu,
                    rest_wavelength_edges_hdu,
                    shifted_wavelength_centers_hdu,
                    shifted_wavelength_edges_hdu,
                    ]
            hdul = fits.HDUList(hdus)
            output_directory = Path(save_directory, line_name)
            make_directory(output_directory)
            savename = Path(output_directory, 'average.fits.gz')
            hdul.writeto(savename, overwrite=True)
            hdul.close()
            return savename

    @staticmethod
    def _set_overlap_to_nan(data: u.Quantity) -> u.Quantity:
        data[np.where(data.value == 0.0)] = np.nan
        keep1 = []
        for col in range(data.shape[1]):
            if ~np.isnan(data[:, col]).all():
                keep1.append(col)
        keep1 = np.array(keep1)
        ind = np.where(~np.isnan(np.sum(data[:, keep1], axis=1)))[0]
        return data[ind]

    @staticmethod
    def _get_data_1d(data_2d,
                     uncertainty_2d,
                     target_mask) -> tuple[u.Quantity, u.Quantity]:
        rows = np.where(np.isnan(np.sum(target_mask, axis=1)))[0]
        calibrated_data_1d = np.sum(data_2d[rows], axis=0)
        calibrated_unc_1d = np.sqrt(
            np.sum(uncertainty_2d[rows]**2, axis=0))
        return calibrated_data_1d, calibrated_unc_1d

    @staticmethod
    def _fit_line(line_wavelengths: u.Quantity,
                  ratios: [float],
                  shifted_wavelengths: u.Quantity,
                  spectrum: u.Quantity,
                  angular_width: u.Quantity) -> ModelResult:
        dwavelength = np.gradient(shifted_wavelengths.value)
        center = np.abs(shifted_wavelengths - line_wavelengths[0]).argmin()
        sigma = angular_width.value * dwavelength[center] / 2.35482

        model = GaussianModel(prefix='peak0_')
        model.set_param_hint(
            'peak0_center',
            min=line_wavelengths.value[0]-sigma,
            max=line_wavelengths.value[0]+sigma)
        model.set_param_hint('peak0_amplitude')
        model.set_param_hint('peak0_sigma', min=0.75*sigma, max=1.25*sigma)
        params = model.make_params(
            center=line_wavelengths.value[0],
            amplitude=10*spectrum.value[center]*dwavelength[center],
            sigma=sigma)
        left = center - 10
        right = center + 10
        if len(line_wavelengths) > 1:
            for i in range(1, len(line_wavelengths)):
                ratio = ratios[i]
                center = np.abs(
                    shifted_wavelengths - line_wavelengths[i]).argmin()
                dwl = (line_wavelengths[i] - line_wavelengths[0]).value
                next_model = GaussianModel(prefix=f'peak{i}_')
                next_model.set_param_hint(f'peak{i}_amplitude',
                                          expr=f'peak0_amplitude*{ratio}',
                                          vary=False)
                next_model.set_param_hint(f'peak{i}_sigma',
                                          expr=f'peak0_sigma',
                                          vary=False)
                next_model.set_param_hint(f'peak{i}_center',
                                          expr=f'peak0_center+{dwl}',
                                          vary=False)
                params += next_model.make_params(
                    center=line_wavelengths.value[i],
                    amplitude=spectrum.value.max(),
                    sigma=sigma)
                model += next_model
                next_left = center - 10
                next_right = center + 10
                if next_left < left:
                    left = next_left
                if next_right > right:
                    right = next_right
        bounds = np.s_[left:right]
        methods = ['powell', 'leastsq', 'least_squares', 'nelder', 'cg']
        fit = None
        for method in methods:
            try:
                fit = model.fit(spectrum.value[bounds], params,
                                x=shifted_wavelengths.value[bounds],
                                method=method)
                unc = fit.params['peak0_amplitude'].stderr
                if unc is not None:
                    break
            except (ValueError, AttributeError):
                continue
        return fit

    @staticmethod
    def _calcualte_brightness_from_1d_fit(fit: ModelResult,
                                          wavelengths: u.Quantity):
        n_models = np.unique([key for key in fit.params.keys()
                              if 'amplitude' in key]).size
        dwavelength = np.gradient(wavelengths.value)
        best_fit = fit.eval(x=wavelengths.value)
        brightness = np.sum(best_fit * dwavelength)
        unc = 0
        for i in range(n_models):
            fit_unc = fit.params[f'peak{i}_amplitude'].stderr
            if (fit_unc is None) or (np.isnan(fit_unc)):
                fit_unc = np.nanstd(fit.data) * np.mean(dwavelength)
            unc = np.sqrt(unc**2 + fit_unc**2)
        # if brightness < 0:
        #     brightness = 0
        return brightness * u.R, unc * u.R

    @staticmethod
    def _align_by_trace(data, unc, traces):
        n, n_spa, n_spe = data.shape
        empty_data = np.zeros((n_spa, n_spe))
        new_data = np.full((n, n_spa*3, n_spe), np.nan)
        new_unc = np.zeros((n, n_spa*3, n_spe))
        count = np.zeros((n_spa*3, n_spe))
        shift_params = dict(order=1, prefilter=False)
        for i in range(n):
            padded_data = np.vstack((empty_data, data[i], empty_data))
            padded_unc = np.vstack((empty_data, unc[i], empty_data))
            padded_count = np.vstack(
                (empty_data, np.ones_like(data[i]), empty_data))
            shifted_data = np.full_like(padded_data, np.nan)
            shifted_unc = np.full_like(padded_unc, np.nan)
            shifted_count = np.zeros_like(padded_data)
            for j in range(n_spe):
                shifted_data[:, j] = shift(
                    padded_data[:, j], -(traces[i]-n_spa/2), **shift_params)
                shifted_unc[:, j] = shift(
                    padded_unc[:, j], -(traces[i]-n_spa/2), **shift_params)
                shifted_count[:, j] = shift(
                    padded_count[:, j], -(traces[i]-n_spa/2), **shift_params)
            shifted_data[shifted_data == 0] = np.nan
            shifted_unc[shifted_unc == 0] = np.nan
            shifted_count[shifted_count > 0] = 1
            new_data[i] = shifted_data
            new_unc[i] = shifted_unc
            count += shifted_count
        new_data = np.nanmean(new_data, axis=0)
        new_unc = np.sqrt(np.nansum(new_unc**2, axis=0)) / count
        ind = ~np.isnan(new_data).all(axis=1)
        return new_data[ind] * u.R / u.nm, new_unc[ind] * u.R / u.nm

    def run(self,
            line_wavelengths: u.Quantity,
            line_name: str,
            line_ratio: list[float],
            exclude: list[int] or dict[str, list[int]] or None):
        """
        Process all individual observations for a set of lines (singlet or
        multiplet).

        Parameters
        ----------
        line_wavelengths : u.Quantity
            Line or set of lines to process.
        line_name : u.Quantity
            The name of the line (will become the directory where the results
            are saved).
        line_ratio : [float]
            Fixed amplitude ratios of line components.
        exclude : [int]
            Indices of observations to exclude from averaging.

        Returns
        -------
        None.
        """
        if exclude is None:
            exclude = []
        elif exclude is dict:
            exclude = exclude[line_name]
        order = self._data.find_order_with_wavelength(line_wavelengths)
        select, select_edges = self._get_data_slice(order, line_wavelengths)
        spatial_scale = self._data.science[0].data_header['SPASCALE']
        spectral_scale = self._data.science[0].data_header['SPESCALE']
        slit_width_bins = self._data.science[0].data_header['SLITWIDB']
        calibration = _FluxCalibration(
            reduced_data_directory=self._reduced_data_directory,
            wavelengths=line_wavelengths, order=order, trim_top=self._trim_top,
            trim_bottom=self._trim_bottom)
        rest_wavelength_selection = (
            self._data.science[0].rest_wavelength_centers[order]
            [select[1]])
        rest_wavelength_edge_selection = (
            self._data.science[0].rest_wavelength_edges[order]
            [select_edges[1]])
        shifted_wavelength_selection = (
            self._data.science[0].doppler_shifted_wavelength_centers[order]
            [select[1]])
        shifted_wavelength_edge_selection = (
            self._data.science[0].doppler_shifted_wavelength_edges[order]
            [select_edges[1]])
        horizontal_positions = self._get_line_indices(
            wavelengths=shifted_wavelength_selection,
            line_wavelengths=line_wavelengths)
        n_traces = len(self._data.trace)
        n_science = len(self._data.science)
        trace_align_average = True
        first_trace_center = None
        if n_traces != n_science:
            msg = (f"      Warning! The number of trace images ({n_traces}) "
                   f"doesn't match the number of science images ({n_science})."
                   f" The pipeline will only use the first trace image.")
            _log(self._save_directory, msg)
            use_trace = self._data.trace[0]
            trace_align_average = False
            traces = []
            for i in range(n_science):
                traces.append(use_trace)
        else:
            traces = self._data.trace
        individual_data_2d = []
        individual_unc_2d = []
        individual_traces = []
        satellite_radii = []
        satellite_sizes = []
        for i, (data, trace) in enumerate(zip(self._data.science, traces)):
            geometry = Geometry(target=data.data_header['TARGET'],
                                observation_time=data.data_header['DATE-OBS'])
            data_selection = self._set_overlap_to_nan(
                deepcopy(data.data[order][select]))
            unc_selection = self._set_overlap_to_nan(
                deepcopy(data.uncertainty[order][select]))
            trace_data_selection = self._set_overlap_to_nan(
                deepcopy(trace.data[order][select]))
            trace_unc_selection = self._set_overlap_to_nan(
                deepcopy(trace.uncertainty[order][select]))
            try:
                trace_fit, trace_fit_unc = _fit_trace(
                    trace_data_selection, trace_unc_selection)
                trace_fit = trace_fit + self._systematic_trace_offset
                if first_trace_center is None:
                    first_trace_center = trace_fit
            except TraceFitError:
                trace_fit = 'error'
                trace_fit_unc = 'error'
            mask = _Mask(data=data_selection, trace_center=trace_fit,
                         horizontal_positions=horizontal_positions,
                         spatial_scale=spatial_scale,
                         spectral_scale=spectral_scale,
                         aperture_radius=self._aperture_radius,
                         satellite_radius=data.angular_radius)

            calibrated_data_2d, calibrated_unc_2d = calibration.calibrate(
                data_selection, unc_selection,
                target_size=mask.satellite_size)

            calibrated_background = _Background(
                data_2d=calibrated_data_2d,
                uncertainty_2d=calibrated_unc_2d,
                rest_wavelengths=rest_wavelength_selection,
                shifted_wavelength_centers=shifted_wavelength_selection,
                shifted_wavelength_edges=shifted_wavelength_edge_selection,
                slit_width_bins=slit_width_bins,
                mask=mask,
                radius=mask.aperture_radius.value,
                spectral_scale=spectral_scale,
                spatial_scale=spatial_scale,
                allow_doppler_shift=self._doppler_shift_background)

            if i not in exclude:
                individual_data_2d.append(calibrated_background.data_2d -
                                          calibrated_background.sky_line_fit)
                individual_unc_2d.append(calibrated_background.uncertainty_2d -
                                         calibrated_background.sky_line_fit)
            if trace_fit != 'error':
                individual_traces.append(trace_fit)
            else:
                individual_traces.append(
                    calibrated_background.data_2d.shape[0] / 2)
            satellite_radii.append(data.angular_radius.value)
            satellite_sizes.append(mask.satellite_size.value)

            calibrated_data_1d, calibrated_unc_1d = self._get_data_1d(
                calibrated_background.bg_sub_data_2d,
                calibrated_background.bg_sub_uncertainty_2d,
                mask.target_mask)

            angular_width = 2 * mask.satellite_radius / spectral_scale
            fit_1d = self._fit_line(line_wavelengths,
                                    line_ratio,
                                    shifted_wavelength_selection,
                                    calibrated_data_1d,
                                    angular_width=angular_width)
            if fit_1d is None:
                continue
            brightness, brightness_unc = (
                self._calcualte_brightness_from_1d_fit(
                    fit_1d, shifted_wavelength_selection))

            file_path = self.save_individual_fits(
                line=line_wavelengths,
                line_name=line_name,
                data_header=data.data_header,
                trace_fit=trace_fit,
                trace_fit_unc=trace_fit_unc,
                geometry=geometry,
                angular_radius=mask.satellite_radius,
                relative_velocity=data.relative_velocity,
                target_masks=mask.target_masks,
                background_masks=mask.background_masks,
                aperture_edges=mask.edges,
                image_data=calibrated_background.data_2d,
                image_data_unc=calibrated_background.uncertainty_2d,
                image_background=calibrated_background.best_fit_2d,
                image_skyline=calibrated_background.sky_line_fit,
                spectrum_1d=calibrated_data_1d,
                spectrum_1d_unc=calibrated_unc_1d,
                best_fit_1d=fit_1d.eval(
                    x=shifted_wavelength_selection.value) * u.R / u.nm,
                best_fit_1d_unc=fit_1d.eval_uncertainty(
                    x=shifted_wavelength_selection.value) * u.R / u.nm,
                wavelength_centers_rest=rest_wavelength_selection,
                wavelength_edges_rest=rest_wavelength_edge_selection,
                wavelength_centers_shifted=shifted_wavelength_selection,
                wavelength_edges_shifted=shifted_wavelength_edge_selection,
                brightness=brightness,
                brightness_unc=brightness_unc,
                save_directory=self._save_directory,
                file_name=data.filename)

            make_quicklook(file_path=file_path)

        # average
        individual_data_2d = np.array(individual_data_2d)
        individual_unc_2d = np.array(individual_unc_2d)
        individual_traces = np.array(individual_traces)
        count = individual_data_2d.shape[0]
        if trace_align_average:
            average_data_2d, average_unc_2d = self._align_by_trace(
                individual_data_2d, individual_unc_2d, individual_traces)
            trace_center = average_data_2d.shape[0] / 2
        else:
            average_data_2d = np.nanmean(
                individual_data_2d, axis=0) * u.R / u.nm
            average_unc_2d = np.sum(
                individual_unc_2d**2, axis=0) / count * u.R / u.nm
            trace_center = first_trace_center
        satellite_radius = np.mean(satellite_radii) * u.arcsec

        mask = _Mask(data=average_data_2d,
                     trace_center=trace_center,
                     horizontal_positions=horizontal_positions,
                     spatial_scale=spatial_scale,
                     spectral_scale=spectral_scale,
                     aperture_radius=self._aperture_radius,
                     satellite_radius=satellite_radius)

        calibrated_background = _Background(
            data_2d=average_data_2d,
            uncertainty_2d=average_unc_2d,
            rest_wavelengths=rest_wavelength_selection,
            shifted_wavelength_centers=shifted_wavelength_selection,
            shifted_wavelength_edges=shifted_wavelength_edge_selection,
            slit_width_bins=slit_width_bins,
            mask=mask,
            radius=mask.aperture_radius.value,
            spectral_scale=spectral_scale,
            spatial_scale=spatial_scale,
            allow_doppler_shift=self._doppler_shift_background,
            fit_skyline=False)

        calibrated_data_1d, calibrated_unc_1d = self._get_data_1d(
            calibrated_background.bg_sub_data_2d,
            calibrated_background.bg_sub_uncertainty_2d,
            mask.target_mask)

        angular_width = 2 * mask.satellite_radius / spectral_scale
        fit_1d = self._fit_line(line_wavelengths,
                                line_ratio,
                                shifted_wavelength_selection,
                                calibrated_data_1d,
                                angular_width=angular_width)
        if fit_1d is None:
            return
        brightness, brightness_unc = (
            self._calcualte_brightness_from_1d_fit(
                fit_1d, shifted_wavelength_selection))

        file_path = self.save_average_fits(
            line=line_wavelengths,
            line_name=line_name,
            data_header=self._data.science[0].data_header,
            tracefit=average_data_2d.shape[0]/2,
            target_masks=mask.target_masks,
            angular_radius=satellite_radius,
            background_masks=mask.background_masks,
            aperture_edges=mask.edges,
            image_data=calibrated_background.data_2d,
            image_data_unc=calibrated_background.uncertainty_2d,
            image_background=calibrated_background.best_fit_2d,
            image_skyline=calibrated_background.sky_line_fit,
            spectrum_1d=calibrated_data_1d,
            spectrum_1d_unc=calibrated_unc_1d,
            best_fit_1d=fit_1d.eval(
                x=shifted_wavelength_selection.value) * u.R / u.nm,
            best_fit_1d_unc=fit_1d.eval_uncertainty(
                x=shifted_wavelength_selection.value) * u.R / u.nm,
            wavelength_centers_rest=rest_wavelength_selection,
            wavelength_edges_rest=rest_wavelength_edge_selection,
            wavelength_centers_shifted=shifted_wavelength_selection,
            wavelength_edges_shifted=shifted_wavelength_edge_selection,
            brightness=brightness,
            brightness_unc=brightness_unc,
            save_directory=self._save_directory)

        make_quicklook(file_path=file_path)


def calibrate_data(reduced_data_directory: str or Path,
                   extended: bool,
                   trim_bottom: int,
                   trim_top: int,
                   aperture_radius: u.Quantity,
                   exclude: [int] = None,
                   skip: [str] = None,
                   systematic_trace_offset: float = 0.0,
                   doppler_shift_background: bool = False):
    """
    This function runs the aurora data calibration pipeline for all wavelengths
    (default O and H or extended Io set).

    Parameters
    ----------
    reduced_data_directory : str or Path
        Absolute path to reduced data from the HIRES pipeline.
    extended : bool
        Whether or not to use the extended wavelength list (for Io).
    trim_bottom : int
        How many rows to trim from the bottom of each order.
    trim_top : int
        How many rows to trim from the top of each order.
    aperture_radius : u.Quantity
        Aperture to use when retrieving brightness in [arcsec].
    exclude : [int]
        Indices of observations to exclude from averaging.
    skip : [str]
        Lines to skip when averaging. Example: `skip=['557.7 nm [O I] ']`.
        Default is None.
    systematic_trace_offset : int or float
        Additional vertical offset for the "trace" in all images.
    doppler_shift_background : bool
        Whether or not to allow the background to Doppler shift. If the slit
        wasn't very long or the signal isn't very strong, this might produce
        bad results and should be turned off.
    """
    if skip is None:
        skip = []
    aurora_lines = AuroraLines(extended=extended)
    log_path = Path(reduced_data_directory.parent, 'calibrated')
    if not log_path.exists():
        log_path.mkdir(parents=True)
    line_data = _LineData(reduced_data_directory=reduced_data_directory,
                          trim_top=trim_top,
                          trim_bottom=trim_bottom,
                          aperture_radius=aperture_radius,
                          systematic_trace_offset=systematic_trace_offset,
                          doppler_shift_background=doppler_shift_background)

    lines = aurora_lines.wavelengths
    line_names = aurora_lines.names
    line_ratios = aurora_lines.ratios

    for (wavelength, name, ratio) in zip(lines, line_names, line_ratios):
        if name in skip:
            _log(log_path, f'   Skipping {name}...')
            continue
        _log(log_path, f'   Calibrating {name} data...')
        try:
            line_data.run(line_wavelengths=wavelength,
                          line_name=name,
                          line_ratio=ratio,
                          exclude=exclude)
        except WavelengthNotFoundError:
            _log(log_path,
                 f'      {name} not captured by HIRES setup! Skipping...')
            continue
        except BackgroundFitError:
            _log(log_path,
                 f'      {name} background fit failed! Skipping...')
            continue
    if exclude is not None:
        _log(log_path, f'Files {exclude} excluded from averaging.')
