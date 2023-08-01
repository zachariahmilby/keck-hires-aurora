import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.table import MaskedColumn
from astropy.time import Time

from hiresaurora.ephemeris import _get_ephemeris
from hiresaurora.general import naif_codes

# noinspection PyUnresolvedReferences
rJ = u.def_unit('R_J', represents=const.R_jup)


class Geometry:
    """
    Class to calcualte observing geometry and plasma-sheet distances.
    """
    def __init__(self, target: str, observation_time: str or Time):
        """
        Parameters
        ----------
        target : str
            The name of the target (options are Jupiter, Io, Europa,  Ganymede,
            Callisto or Maunakea).
        observation_time : str or Time
            Time of the observation on Earth.
        """
        self._target = target
        self._observation_time = Time(observation_time, format='isot')
        self._corrected_time = self._remove_light_travel_time()
        self._coordinates = self._get_subjovian_coordinates()
        self._observing_geometry = self._get_observing_geometry()

    def __str__(self) -> str:
        print_strs = ['Observational Geometry:',
                      f'   Observation time: {self._observation_time}',
                      f'   Light-corrected time: {self._corrected_time}',
                      f'   Target magnetic latitude: '
                      f'{np.round(self.magnetic_latitude, 4)}',
                      f'   Target magnetic longitude: '
                      f'{np.round(self.magnetic_longitude, 4)}',
                      f'   Target orbital distance: '
                      f'{np.round(self.orbital_distance.to(rJ), 4)}',
                      f'   Distance from centrifugal equator: '
                      f'{np.round(self.height, 4)}',
                      f'   Sub-observer latitude: '
                      f'{np.round(self.sub_observer_latitude, 4)}',
                      f'   Sub-observer longitude: '
                      f'{np.round(self.sub_observer_longitude, 4)}']
        return '\n'.join(print_strs)

    # noinspection PyUnresolvedReferences
    @staticmethod
    def _centrifugal_equator_latitude(
            r: u.Quantity, lon: u.Quantity = 159 * u.degree) -> u.Quantity:
        """
        Phipps and Bagenal (2021) equation (2). Returns centrifugal equator
        latitude in degrees.
        """
        a = (1.66 * u.degree).to(u.radian)
        b = 0.131 * u.radian
        c = 1.62 * u.radian
        d = (7.76 * u.degree).to(u.radian)
        e = (249 * u.degree).to(u.radian)

        r = r.si
        lon = lon.to(u.radian)

        return ((a * np.tanh(b * (r / const.R_jup).value - c) + d) * np.sin(
            lon - e)).to(u.degree)

    @staticmethod
    def _convert_to_height(r: u.Quantity, lat: u.Quantity) -> u.Quantity:
        """
        Calculate height above or below centrifugal equator in Jupiter radii
        using right-triangle geometry.
        """
        lat = lat.to(u.radian)
        return r.to(rJ) * np.tan(-lat)

    @staticmethod
    def _convert_west_longitude_to_east(lon: u.Quantity) -> u.Quantity:
        """
        Convert longitude in degrees from west longitude to east longitude.
        """
        return 360 * u.degree - lon.to(u.degree)

    @staticmethod
    def _convert_to_quantity(
            quantity: MaskedColumn, si: bool = True) -> u.Quantity:
        """
        Convert Horizons ephemeris output to a single astropy quantity with
        units. By default it will also convert to SI units.
        """
        value = quantity.value[0]
        unit = quantity.unit
        if si:
            return (value * unit).si
        else:
            return value * unit

    def _remove_light_travel_time(self) -> Time:
        """
        Adjust observation time to account for light travel time between target
        satellite and Maunakea.
        """
        eph = _get_ephemeris(target=self._target,
                             time=self._observation_time,
                             location=naif_codes['Maunakea'])
        dt = self._convert_to_quantity(eph['lighttime'])
        return self._observation_time - dt

    def _get_subjovian_coordinates(self) -> dict:
        """
        Need to account for the time in the Jovian system when the observations
        we captured actually took place.
        """
        eph = _get_ephemeris(target='Jupiter', time=self._corrected_time,
                             location=f'@{naif_codes[self._target]}')
        subjovian_latitude = self._convert_to_quantity(eph['PDObsLat'])
        subjovian_longitude = self._convert_west_longitude_to_east(
            self._convert_to_quantity(eph['PDObsLon']))
        distance_from_jupiter = self._convert_to_quantity(eph['delta'])
        equator_latitude = self._centrifugal_equator_latitude(
            r=distance_from_jupiter, lon=subjovian_longitude)
        latitude = subjovian_latitude - equator_latitude
        height = self._convert_to_height(
            r=distance_from_jupiter,
            lat=equator_latitude + subjovian_latitude)
        return {
            'magnetic_latitude': latitude.to(u.degree),
            'magnetic_longitude': subjovian_longitude,
            'distance': distance_from_jupiter.to(rJ),
            'height': height.to(rJ)
        }

    def _get_observing_geometry(self) -> dict:
        """
        Horizons takes light travel time into account, so this calculation
        doesn't need any time modification; it will return the sub-observer
        coordinates as observed.
        """
        eph = _get_ephemeris(target=self._target, time=self._observation_time,
                             location=naif_codes['Maunakea'])
        sub_observer_latitude = self._convert_to_quantity(eph['PDObsLat']).to(
            u.degree)
        sub_observer_longitude = self._convert_west_longitude_to_east(
            self._convert_to_quantity(eph['PDObsLon'])).to(u.degree)
        distance = self._convert_to_quantity(eph['delta']).to(u.km)
        north_pole_angle = self._convert_to_quantity(eph['NPole_ang'])
        return {
            'sub_observer_latitude': sub_observer_latitude,
            'sub_observer_longitude': sub_observer_longitude,
            'distance': distance,
            'north_pole_angle': north_pole_angle
        }

    @property
    def observation_time(self) -> Time:
        return self._observation_time

    @property
    def light_corrected_time(self) -> Time:
        return self._corrected_time

    @property
    def magnetic_latitude(self) -> u.Quantity:
        return self._coordinates['magnetic_latitude']

    @property
    def magnetic_longitude(self) -> u.Quantity:
        return self._coordinates['magnetic_longitude']

    @property
    def orbital_distance(self) -> u.Quantity:
        return self._coordinates['distance']

    @property
    def height(self) -> u.Quantity:
        return self._coordinates['height']

    @property
    def sub_observer_latitude(self) -> u.Quantity:
        return self._observing_geometry['sub_observer_latitude']

    @property
    def sub_observer_longitude(self) -> u.Quantity:
        return self._observing_geometry['sub_observer_longitude']

    @property
    def distance(self) -> u.Quantity:
        return self._observing_geometry['distance']

    @property
    def north_pole_angle(self) -> u.Quantity:
        return self._observing_geometry['north_pole_angle']
