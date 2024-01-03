import warnings

import astropy.units as u
import numpy as np


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
        if trace_center == 'error':
            trace_center = data.shape[0] / 2
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

    @property
    def aperture_size_pix(self) -> int:
        return np.where(np.isnan(self._masks['target_mask'].flatten()))[0].size
