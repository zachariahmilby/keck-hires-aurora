import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.time import Time
from hirespipeline.files import make_directory
from lmfit.models import PolynomialModel, GaussianModel, ConstantModel

from hiresaurora.background_subtraction import _Background
from hiresaurora.calibration import _FluxCalibration
from hiresaurora.ephemeris import _get_ephemeris
from hiresaurora.general import _doppler_shift_wavelengths, AuroraLines
from hiresaurora.graphics import make_quicklook


def _fit_trace(data: u.Quantity or np.ndarray,
               unc: u.Quantity or np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Calculate fractional pixel vertical position of the trace.
    """
    gaussian_model = GaussianModel()
    constant_model = ConstantModel()
    composite_model = gaussian_model + constant_model
    x = np.arange(data.shape[0])
    centers = np.zeros(data.shape[1])
    centers_unc = np.zeros_like(centers)
    for i in range(data.shape[1]):
        column = data[:, i]
        weights = 1 / unc[:, i] ** 2
        if isinstance(data, u.Quantity):
            column = column.value
            weights = weights.value
        gaussian_params = gaussian_model.guess(column, x=x)
        constant_params = constant_model.guess(column, x=x)
        composite_params = gaussian_params + constant_params
        fit = composite_model.fit(
            column, composite_params, weights=weights, x=x)
        centers[i] = fit.params['center'].value
        centers_unc[i] = fit.params['center'].stderr
    model = PolynomialModel(degree=0)
    x = np.arange(data.shape[1])
    params = model.guess(centers, x=x)
    fit = model.fit(centers, params, weights=1 / centers_unc ** 2, x=x)
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
        target = self._data_header['OBJECT']
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
    def science(self) -> [_ImageData]:
        return self._science_data

    @property
    def trace(self) -> [_ImageData]:
        return self._trace_data


class _Mask:
    """
    Class to generate aperture masks.
    """
    def __init__(self, data: u.Quantity, trace_center: float,
                 horizontal_positions: [float],
                 horizontal_offset: int or float, spatial_scale: float,
                 spectral_scale: float, aperture_radius: u.Quantity,
                 satellite_radius: u.Quantity):
        """
        Parameters
        ----------
        data : u.Quantity
            The data.
        trace_center : float
            The fractional vertical pixel position of the trace.
        horizontal_positions : [float]
            The fractional horizontal pixel position(s) of the emission lines.
        horizontal_offset : int or float
            Any additional offset if the wavelength solution is off.
        spatial_scale : float
            The spatial scale, probably in [arcsec/bin] since it's read from
            the reduced FITS header.
        spectral_scale : float
            The spectral scale, probably in [arcsec/bin] since it's read from
            the reduced FITS header.
        aperture_radius : u.Quantity
            The radius of the extraction aperture in angular units (like
            arcsec).
        satellite_radius : u.Quantity
            The angular radius of the target satellite.
        """
        self._data = data
        self._trace_center = trace_center
        self._horizontal_positions = horizontal_positions
        self._horizontal_offset = horizontal_offset
        self._spatial_scale = spatial_scale
        self._spectral_scale = spectral_scale
        self._aperture_radius = aperture_radius.to(u.arcsec)
        self._satellite_radius = satellite_radius.to(u.arcsec)
        self._masks = self._make_masks()

    def _make_masks(self):
        """
        Make a target mask (isolating the background) and background mask
        (isolating the target).
        """
        shape = self._data.shape
        x, y = np.meshgrid(np.arange(shape[1]) * self._spectral_scale,
                           np.arange(shape[0]) * self._spatial_scale)
        target_masks = []
        background_masks = []
        edges = []
        if self._trace_center is None:
            self._trace_center = shape[0] / 2
        for horizontal_position in self._horizontal_positions:
            horizontal_position += self._horizontal_offset
            distance = np.sqrt(
                (x - self._spectral_scale * horizontal_position) ** 2 +
                (y - self._spatial_scale * self._trace_center) ** 2)
            mask = np.ones_like(distance)
            mask[np.where(distance < self._aperture_radius.value)] = np.nan
            mask[np.where(distance >= self._aperture_radius.value)] = 1
            target_masks.append(mask)
            mask = np.ones_like(distance)
            mask[np.where(distance < self._aperture_radius.value)] = 1
            mask[np.where(distance >= self._aperture_radius.value)] = np.nan
            background_masks.append(mask)
            edge = np.zeros_like(mask)
            edge[np.where(distance < self._aperture_radius.value)] = 1
            edges.append(edge)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            target_mask = np.mean(target_masks, axis=0)
            background_mask = np.nanmean(background_masks, axis=0)
        edges = np.sum(edges, axis=0)
        edges[np.where(edges > 0)] = 1
        return {'target_mask': target_mask,
                'target_masks': np.array(target_masks),
                'background_mask': background_mask,
                'background_masks': np.array(background_masks),
                'edges': edges, 'x': x, 'y': y}

    @property
    def target_mask(self) -> np.ndarray:
        return self._masks['target_mask']

    @property
    def target_masks(self) -> np.ndarray:
        return self._masks['target_masks']

    @property
    def background_mask(self) -> np.ndarray:
        return self._masks['background_mask']

    @property
    def background_masks(self) -> np.ndarray:
        return self._masks['background_masks']

    @property
    def edges(self) -> np.ndarray:
        return self._masks['edges']

    @property
    def x(self) -> np.ndarray:
        return self._masks['x']

    @property
    def y(self) -> np.ndarray:
        return self._masks['y']

    @property
    def horizontal_positions(self) -> np.ndarray:
        return self._horizontal_positions + self._horizontal_offset

    @property
    def vertical_position(self) -> float:
        return self._trace_center

    @property
    def aperture_radius(self) -> u.Quantity:
        return self._aperture_radius

    @property
    def aperture_size(self) -> u.Quantity:
        return np.pi * self._aperture_radius ** 2

    @property
    def satellite_radius(self) -> u.Quantity:
        return self._satellite_radius

    @property
    def satellite_size(self) -> u.Quantity:
        return np.pi * self._satellite_radius ** 2

    @property
    def pixel_size(self) -> u.Quantity:
        return self._spectral_scale * self._spectral_scale * u.arcsec**2


# noinspection DuplicatedCode
class _LineData:
    """
    Process individual data for a given emisison line or multiplet.
    """
    def __init__(self, reduced_data_directory: str or Path,
                 aperture_radius: u.Quantity, average_aperture_scale: float,
                 horizontal_offset: int or float = 0.0, trim_top: int = 2,
                 trim_bottom: int = 2, interpolate_aperture: bool = False,
                 fill_holes: bool = False):
        """
        Parameters
        ----------
        reduced_data_directory : str or Path
            File path to the reduced data directory from the HIRES data
            reduction pipeline.
        aperture_radius : u.Quantity
            The extraction aperture radius in arcsec.
        average_aperture_scale : float
            Factor to scale the aperture radius by for the average images.
        horizontal_offset : int or float
            Any additional offset if the wavelength solution is off.
        trim_top : int
            Number of additional rows to trim off the top of the rectified
            data (the background fitting significantly improves if the sawtooth
            slit edges are excluded). The default is 2.
        trim_bottom : int
            Number of additional rows to trim off of the bottom of the
            rectified data (the background fitting significantly improves if
            the sawtooth slit edges are excluded). The default is 2.
        interpolate_aperture : bool
            If the aperture location appears to be a hole or hill, you can
            replace it with an interpolation based on the surrounding column
            values.
        fill_holes : bool
            If there are bad "holes" left over from background subtraction,
            this will attempt to remove them.
        """
        self._reduced_data_directory = Path(reduced_data_directory)
        self._aperture_radius = aperture_radius.to(u.arcsec)
        self._average_aperture_scale = average_aperture_scale
        self._horizontal_offset = horizontal_offset
        self._trim_top = trim_top
        self._trim_bottom = trim_bottom
        self._data = _RawData(self._reduced_data_directory)
        self._save_directory = self._parse_save_directory()
        self._interpolate_aperture = interpolate_aperture
        self._fill_holes = fill_holes

    def _parse_save_directory(self) -> Path:
        return Path(self._reduced_data_directory.parent, 'calibrated')

    def _get_data_slice(self, order: int, line_wavelengths: u.Quantity,
                        dwavelength: u.Quantity = 0.25 * u.nm
                        ) -> (np.s_, np.s_):
        """
        Get slice corresponding to ±dwavelength from Doppler-shifted
        wavelength. Uses minimum and maximum of entire observing sequence to
        account for variation in the bounds due to changing Doppler shift.
        Makes slices for both data and wavelength edges (one additional entry
        along horizontal axis).
        """
        lefts = []
        rights = []
        for data in self._data.science:
            wavelengths = data.doppler_shifted_wavelength_centers[order]
            left_bound = np.min(line_wavelengths) - dwavelength
            left = np.abs(wavelengths - left_bound).argmin()
            if left < 0:
                left = 0
            right_bound = np.max(line_wavelengths) + dwavelength
            right = np.abs(wavelengths - right_bound).argmin() + 1
            if right > wavelengths.shape[0]:
                right = wavelengths.shape[0]
            lefts.append(left)
            rights.append(right)
        if self._trim_top == 0:
            top = self._data.science[0].data.shape[0]
        else:
            top = -self._trim_top
        data_slice = np.s_[self._trim_bottom:top, np.min(lefts):np.max(rights)]
        edge_slice = np.s_[self._trim_bottom:-top,
                           np.min(lefts):np.max(rights)+1]
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
                           'DATE-OBS', 'OBJECT', 'EXPTIME', 'IMAGETYP',
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
                           'NAXIS2', 'GCOUNT', 'PCOUNT', 'BUNIT', 'OBJECT',
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
    def _set_primary_header(primary_hdu: fits.PrimaryHDU,
                            line_name: str,
                            brightness: u.Quantity,
                            brightness_unc: u.Quantity,
                            brightness_std: u.Quantity,
                            trace: float = None, trace_unc: float = None):
        """
        Set primary extension header information (not observation-specific).
        """
        del primary_hdu.header['BUNIT']
        del primary_hdu.header['OBJECT']
        del primary_hdu.header['AIRMASS']
        del primary_hdu.header['DATE-OBS']
        del primary_hdu.header['EXPTIME']
        primary_hdu.header.set('LINE', f'{line_name}',
                               'targeted emission line')
        if trace is not None:
            primary_hdu.header.set('TRACEFIT', trace,
                                   'trace fit fractional pixel')
        if trace_unc is not None:
            primary_hdu.header.set('TRACEUNC', trace_unc,
                                   'trace fit fractional pixel uncertainty')
        primary_hdu.header.set(
            'BGHTNESS', brightness.value,
            f'retrieved brightness [{brightness.unit}]')
        primary_hdu.header.set(
            'BGHT_UNC', brightness_unc.value,
            f'retrieved brightness uncertainty [{brightness_unc.unit}]')
        primary_hdu.header.set(
            'BGHT_STD', brightness_std.value,
            f'retrieved brightness standard deviation [{brightness_std.unit}]')

    # noinspection DuplicatedCode
    def save_individual_fits(
            self, line: u.Quantity, line_name: str,
            data_header: fits.Header, trace_header: fits.Header,
            raw_data: u.Quantity, raw_unc: u.Quantity,
            trace_data: u.Quantity, trace_unc: u.Quantity,
            trace_fit: float, trace_fit_unc: float,
            background: u.Quantity, background_unc: u.Quantity,
            target_masks: np.ndarray, background_masks: np.ndarray,
            aperture_edges: np.ndarray,
            calibrated_data: u.Quantity, calibrated_unc: u.Quantity,
            wavelength_centers: u.Quantity, wavelength_edges: u.Quantity,
            brightness: u.Quantity, brightness_unc: u.Quantity,
            brightness_std: u.Quantity,
            save_directory: Path, file_name: str):
        """
        Save calibrated data for an individual observation as a FITS file.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', category=fits.verify.VerifyWarning)

            primary_hdu = fits.PrimaryHDU(header=data_header.copy())
            self._set_primary_header(primary_hdu=primary_hdu,
                                     line_name=line_name,
                                     brightness=brightness,
                                     brightness_unc=brightness_unc,
                                     brightness_std=brightness_std,
                                     trace=trace_fit, trace_unc=trace_fit_unc)
            targeted_lines_hdu = self._make_image_hdu(data=line, header=None,
                                                      name='TARGETED_LINES')
            raw_hdu = self._make_image_hdu(
                data=raw_data, header=data_header.copy(), name='RAW')
            raw_unc_hdu = self._make_image_hdu(
                data=raw_unc, header=data_header.copy(), name='RAW_UNC')
            trace_hdu = self._make_image_hdu(
                data=trace_data, header=trace_header.copy(), name='TRACE')
            trace_unc_hdu = self._make_image_hdu(
                data=trace_unc, header=trace_header.copy(), name='TRACE_UNC')
            target_mask_hdu = self._make_image_hdu(
                data=target_masks, header=None, name='TARGET_MASKS',
                comment='Mask(s) used to isolate background when calculating '
                        'background fit.')
            background_mask_hdu = self._make_image_hdu(
                data=background_masks, header=None, name='BACKGROUND_MASKS',
                comment='Mask(s) used to isolate target when calculating '
                        'emission brightness.')
            aperture_edges_hdu = self._make_image_hdu(
                data=aperture_edges, header=None, name='APERTURE_EDGES')
            background_fit_hdu = self._make_image_hdu(
                data=background, header=data_header.copy(),
                name='BACKGROUND_FIT')
            background_fit_unc_hdu = self._make_image_hdu(
                data=background_unc, header=data_header.copy(),
                name='BACKGROUND_FIT_UNC')
            calibrated_hdu = self._make_image_hdu(
                data=calibrated_data, header=data_header.copy(),
                name='CALIBRATED')
            calibrated_unc_hdu = self._make_image_hdu(
                data=calibrated_unc, header=data_header.copy(),
                name='CALIBRATED_UNC')
            wavelength_centers_hdu = self._make_image_hdu(
                data=wavelength_centers, header=None,
                name='WAVELENGTH_CENTERS',
                comment='Doppler-shifted wavelengths.')
            wavelength_edges_hdu = self._make_image_hdu(
                data=wavelength_edges, header=None, name='WAVELENGTH_EDGES',
                comment='Doppler-shifted wavelengths.')

            hdus = [primary_hdu, targeted_lines_hdu, raw_hdu, raw_unc_hdu,
                    trace_hdu, trace_unc_hdu, target_mask_hdu,
                    background_mask_hdu, aperture_edges_hdu,
                    background_fit_hdu, background_fit_unc_hdu,
                    calibrated_hdu, calibrated_unc_hdu, wavelength_centers_hdu,
                    wavelength_edges_hdu]
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
            self, line: u.Quantity, line_name: str,
            data_header: fits.Header,
            raw_data: u.Quantity, raw_unc: u.Quantity,
            tracefit: float,
            background: u.Quantity, background_unc: u.Quantity,
            target_masks: np.ndarray, background_masks: np.ndarray,
            aperture_edges: np.ndarray,
            calibrated_data: u.Quantity, calibrated_unc: u.Quantity,
            wavelength_centers: u.Quantity, wavelength_edges: u.Quantity,
            brightness: u.Quantity, brightness_unc: u.Quantity,
            brightness_std: u.Quantity,
            save_directory: Path):
        """
        Save calibrated data for an average observation as a FITS file.
        """
        with warnings.catch_warnings():
            warnings.simplefilter(
                'ignore', category=fits.verify.VerifyWarning)

            primary_hdu = fits.PrimaryHDU(header=data_header.copy())
            self._set_primary_header(primary_hdu=primary_hdu,
                                     line_name=line_name,
                                     brightness=brightness,
                                     brightness_unc=brightness_unc,
                                     brightness_std=brightness_std,
                                     trace=tracefit)
            targeted_lines_hdu = self._make_image_hdu(data=line,
                                                      header=None,
                                                      name='TARGETED_LINES')
            raw_hdu = self._make_image_hdu(
                data=raw_data, header=data_header.copy(), name='RAW')
            raw_unc_hdu = self._make_image_hdu(
                data=raw_unc, header=data_header.copy(),
                name='RAW_UNC')
            target_mask_hdu = self._make_image_hdu(
                data=target_masks, header=None, name='TARGET_MASKS',
                comment='Mask(s) used to isolate background when calculating '
                        'background fit.')
            background_mask_hdu = self._make_image_hdu(
                data=background_masks, header=None,
                name='BACKGROUND_MASKS',
                comment='Mask(s) used to isolate target when calculating '
                        'emission brightness.')
            aperture_edges_hdu = self._make_image_hdu(
                data=aperture_edges, header=None,
                name='APERTURE_EDGES')
            background_fit_hdu = self._make_image_hdu(
                data=background, header=data_header.copy(),
                name='BACKGROUND_FIT')
            background_fit_unc_hdu = self._make_image_hdu(
                data=background_unc, header=data_header.copy(),
                name='BACKGROUND_FIT_UNC')
            calibrated_hdu = self._make_image_hdu(
                data=calibrated_data, header=data_header.copy(),
                name='CALIBRATED')
            calibrated_unc_hdu = self._make_image_hdu(
                data=calibrated_unc, header=data_header.copy(),
                name='CALIBRATED_UNC')
            wavelength_centers_hdu = self._make_image_hdu(
                data=wavelength_centers, header=None,
                name='WAVELENGTH_CENTERS',
                comment='Doppler-shifted wavelengths.')
            wavelength_edges_hdu = self._make_image_hdu(
                data=wavelength_edges, header=None,
                name='WAVELENGTH_EDGES',
                comment='Doppler-shifted wavelengths.')

            hdus = [primary_hdu, targeted_lines_hdu,
                    raw_hdu, raw_unc_hdu,
                    target_mask_hdu, background_mask_hdu, aperture_edges_hdu,
                    background_fit_hdu, background_fit_unc_hdu,
                    calibrated_hdu, calibrated_unc_hdu,
                    wavelength_centers_hdu, wavelength_edges_hdu]
            hdul = fits.HDUList(hdus)
            output_directory = Path(save_directory, line_name)
            make_directory(output_directory)
            savename = Path(output_directory, 'average.fits.gz')
            hdul.writeto(savename, overwrite=True)
            hdul.close()
            return savename

    # noinspection DuplicatedCode
    def run_individual(self, line_wavelengths: u.Quantity, line_name: str):
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

        Returns
        -------
        None.
        """
        order = self._data.find_order_with_wavelength(line_wavelengths)

        select, select_edges = self._get_data_slice(order, line_wavelengths)
        spatial_scale = self._data.science[0].data_header['SPASCALE']
        spectral_scale = self._data.science[0].data_header['SPESCALE']
        calibration = _FluxCalibration(
            reduced_data_directory=self._reduced_data_directory,
            wavelengths=line_wavelengths, order=order, trim_top=self._trim_top,
            trim_bottom=self._trim_bottom)
        wavelength_selection = (
            self._data.science[0].doppler_shifted_wavelength_centers[order]
            [select[1]])
        wavelength_edge_selection = (
            self._data.science[0].doppler_shifted_wavelength_edges[order]
            [select_edges[1]])
        for data, trace in zip(self._data.science, self._data.trace):
            data_selection = data.data[order][select]
            unc_selection = data.uncertainty[order][select]
            trace_data_selection = trace.data[order][select]
            trace_unc_selection = trace.uncertainty[order][select]
            trace_fit, trace_fit_unc = _fit_trace(
                trace_data_selection, trace_unc_selection)
            horizontal_positions = self._get_line_indices(
                wavelengths=wavelength_selection,
                line_wavelengths=line_wavelengths)
            mask = _Mask(data=data_selection, trace_center=trace_fit,
                         horizontal_positions=horizontal_positions,
                         horizontal_offset=self._horizontal_offset,
                         spatial_scale=spatial_scale,
                         spectral_scale=spectral_scale,
                         aperture_radius=self._aperture_radius,
                         satellite_radius=data.angular_radius)
            background = _Background(
                data=data_selection,
                uncertainty=unc_selection,
                mask=mask.target_mask,
                radius=mask.aperture_radius.value,
                spectral_scale=spectral_scale,
                spatial_scale=spatial_scale,
                interpolate_aperture=self._interpolate_aperture,
                fill_holes=self._fill_holes)
            calibrated_data, calibrated_unc = calibration.calibrate(
                background.data, background.uncertainty,
                target_size=mask.satellite_size)
            dwavelength = np.repeat(
                np.gradient(wavelength_selection)[None, :],
                data_selection.shape[0], axis=0)
            calibrated_data = calibrated_data * dwavelength
            calibrated_unc = calibrated_unc * dwavelength
            brightness = np.nansum(
                calibrated_data.value * mask.background_mask
            ) * calibrated_data.unit
            qty = (calibrated_unc * mask.background_mask)**2
            brightness_unc = np.sqrt(np.nansum(qty))
            n = np.count_nonzero(np.isnan(mask.target_mask))
            brightness_std = (np.nanstd(calibrated_data * mask.target_mask)
                              * np.sqrt(n))

            # calibrate data to "per pixel" assuming emission from disk the
            # size of the target satellite
            calibrated_data *= mask.satellite_size / mask.pixel_size
            calibrated_unc *= mask.satellite_size / mask.pixel_size

            file_path = self.save_individual_fits(
                line=line_wavelengths,
                line_name=line_name,
                data_header=data.data_header,
                trace_header=trace.data_header,
                raw_data=data_selection,
                raw_unc=unc_selection,
                trace_data=trace_data_selection,
                trace_unc=trace_unc_selection,
                trace_fit=trace_fit,
                trace_fit_unc=trace_fit_unc,
                background=background.best_fit,
                background_unc=background.best_fit_uncertainty,
                target_masks=mask.target_masks,
                background_masks=mask.background_masks,
                aperture_edges=mask.edges,
                calibrated_data=calibrated_data,
                calibrated_unc=calibrated_unc,
                wavelength_centers=wavelength_selection,
                wavelength_edges=wavelength_edge_selection,
                brightness=np.round(brightness, 4),
                brightness_unc=np.round(brightness_unc, 4),
                brightness_std=np.round(brightness_std, 4),
                save_directory=self._save_directory,
                file_name=data.filename)

            make_quicklook(file_path=file_path)

    # noinspection DuplicatedCode
    def run_average(self, line_wavelengths: u.Quantity, line_name: str,
                    exclude: [int], trace_offset: int or float = 0.0):
        """
        Process average of all individual observations for a set of lines
        (singlet or multiplet).

        Parameters
        ----------
        line_wavelengths : u.Quantity
            Line or set of lines to process.
        line_name : u.Quantity
            The name of the line (will become the directory where the results
            are saved).
        exclude : [int]
            Observations to exclude from averaging.
        trace_offset : int or float
            Additional vertical offset for "trace" in the average image.

        Returns
        -------
        None.
        """
        order = self._data.find_order_with_wavelength(line_wavelengths)
        select, select_edges = self._get_data_slice(order, line_wavelengths)
        data_header = self._data.science[0].data_header
        spatial_scale = data_header['SPASCALE']
        spectral_scale = data_header['SPESCALE']
        unit = self._data.science[0].data.unit
        angular_radius = self._data.science[0].angular_radius

        if exclude is not None:
            use_data = np.array([i for i in np.arange(len(self._data.science))
                                 if i not in exclude])
        else:
            use_data = np.arange(len(self._data.science))

        calibration = _FluxCalibration(
            reduced_data_directory=self._reduced_data_directory,
            wavelengths=line_wavelengths, order=order, trim_top=self._trim_top,
            trim_bottom=self._trim_bottom)

        wavelength_selection = (
            self._data.science[0].doppler_shifted_wavelength_centers[order]
            [select[1]])
        wavelength_edge_selection = (
            self._data.science[0].doppler_shifted_wavelength_edges[order]
            [select_edges[1]])
        data_selection = np.nanmean(
            [data.data[order][select].value
             for data in np.array(self._data.science)[use_data]],
            axis=0) * unit
        unc_selection = np.sqrt(
            np.nansum([data.uncertainty[order][select].value ** 2
                       for data in np.array(self._data.science)[use_data]],
                      axis=0)) * unit / len(use_data)
        trace_fit = (data_selection.shape[0] - 1) / 2 + trace_offset
        horizontal_positions = self._get_line_indices(
            wavelengths=wavelength_selection,
            line_wavelengths=line_wavelengths)
        mask = _Mask(data=data_selection, trace_center=trace_fit,
                     horizontal_positions=horizontal_positions,
                     horizontal_offset=self._horizontal_offset,
                     spatial_scale=spatial_scale,
                     spectral_scale=spectral_scale,
                     aperture_radius=(self._aperture_radius *
                                      self._average_aperture_scale),
                     satellite_radius=angular_radius)
        background = _Background(
            data=data_selection,
            uncertainty=unc_selection,
            mask=mask.target_mask,
            radius=mask.aperture_radius.value,
            spectral_scale=spectral_scale,
            spatial_scale=spatial_scale,
            interpolate_aperture=self._interpolate_aperture,
            fill_holes=self._fill_holes)
        calibrated_data, calibrated_unc = calibration.calibrate(
            background.data, background.uncertainty,
            target_size=mask.satellite_size)
        dwavelength = np.repeat(
            np.gradient(wavelength_selection)[None, :],
            data_selection.shape[0], axis=0)
        calibrated_data = calibrated_data * dwavelength
        calibrated_unc = calibrated_unc * dwavelength
        brightness = np.nansum(calibrated_data.value * mask.background_mask
                               ) * calibrated_data.unit
        qty = (calibrated_unc * mask.background_mask) ** 2
        brightness_unc = np.sqrt(np.nansum(qty))
        n = np.count_nonzero(~np.isnan(mask.target_mask))
        brightness_std = (np.nanstd(calibrated_data * mask.target_mask)
                          * np.sqrt(n))

        # calibrate data to "per pixel" assuming emission from disk the
        # size of the target satellite
        calibrated_data *= mask.satellite_size / mask.pixel_size
        calibrated_unc *= mask.satellite_size / mask.pixel_size

        file_path = self.save_average_fits(
            line=line_wavelengths,
            line_name=line_name,
            data_header=data_header,
            raw_data=data_selection,
            raw_unc=unc_selection,
            tracefit=trace_fit,
            background=background.best_fit,
            background_unc=background.best_fit_uncertainty,
            target_masks=mask.target_masks,
            background_masks=mask.background_masks,
            aperture_edges=mask.edges,
            calibrated_data=calibrated_data,
            calibrated_unc=calibrated_unc,
            wavelength_centers=wavelength_selection,
            wavelength_edges=wavelength_edge_selection,
            brightness=np.round(brightness, 4),
            brightness_unc=np.round(brightness_unc, 4),
            brightness_std=np.round(brightness_std, 4),
            save_directory=self._save_directory)

        make_quicklook(file_path=file_path)


def calibrate_data(reduced_data_directory: str or Path, extended: bool,
                   trim_bottom: int, trim_top: int,
                   aperture_radius: u.Quantity,
                   average_aperture_scale: float = 1.0,
                   horizontal_offset: int = 0, exclude: [int] = None,
                   average_trace_offset: int or float = 0.0,
                   interpolate_aperture: bool = False,
                   fill_holes: bool = False):
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
    average_aperture_scale : float
        If you want to scale the aperture radius for the averages.
    horizontal_offset : int or float
        Additional pixel offset for horizontal centering if wavelength solution
        is off.
    exclude : [int]
        Indices of observations to exclude from averaging.
    average_trace_offset : int or float
        Additional vertical offset for "trace" in the average image.
    interpolate_aperture : bool
        If the aperture location appears to be a hole or hill, you can replace
        it with an interpolation based on the surrounding column values.
    fill_holes : bool
            If there are bad "holes" left over from background subtraction,
            this will attempt to remove them.
    """

    aurora_lines = AuroraLines(extended=extended)
    line_data = _LineData(reduced_data_directory=reduced_data_directory,
                          horizontal_offset=horizontal_offset,
                          trim_top=trim_top,
                          trim_bottom=trim_bottom,
                          aperture_radius=aperture_radius,
                          average_aperture_scale=average_aperture_scale,
                          interpolate_aperture=interpolate_aperture,
                          fill_holes=fill_holes)

    lines = aurora_lines.wavelengths
    line_names = aurora_lines.names
    for line_wavelengths, line_name in zip(lines, line_names):
        print(f'   Calibrating {line_name} data...')
        try:
            line_data.run_individual(line_wavelengths=line_wavelengths,
                                     line_name=line_name)
            line_data.run_average(line_wavelengths=line_wavelengths,
                                  line_name=line_name, exclude=exclude,
                                  trace_offset=average_trace_offset)
        except ValueError:
            print(f'   Failed to calibrate {line_name} data...')
            continue
        except WavelengthNotFoundError:
            print(f'      {line_name} not captured by HIRES setup! '
                  f'Skipping...')
            continue
    if exclude is not None:
        print(f'Files {exclude} excluded from averaging.')
