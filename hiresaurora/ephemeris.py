import numpy as np
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from hiresaurora.general import naif_codes


def _horizons_query(target: str, epochs: float or dict,
                    location: str = naif_codes['Maunakea'],
                    airmass_lessthan: int | float | None = 2,
                    skip_daylight: bool = False) -> dict:
    """
    Wrapper code for a typical Horizons query.
    """
    obj = Horizons(id=naif_codes[target], location=location, epochs=epochs)
    if airmass_lessthan is None:
        return obj.ephemerides(skip_daylight=skip_daylight)
    else:
        return obj.ephemerides(airmass_lessthan=airmass_lessthan,
                               skip_daylight=skip_daylight)


def _get_ephemeris(target: str, time: Time or str,
                   location: str = naif_codes['Maunakea'],
                   airmass_lessthan: int | float | None = 3,
                   skip_daylight: bool = False) -> dict:
    """
    Query the JPL Horizons System for a single time.
    """
    epochs = Time(time).jd
    return _horizons_query(target=target, location=location,
                           epochs=epochs, airmass_lessthan=airmass_lessthan,
                           skip_daylight=skip_daylight)


def _get_ephemerides(target: str, starting_datetime: str, ending_datetime: str,
                     step: str = '1m', location: str = naif_codes['Maunakea'],
                     airmass_lessthan: int | float | None = 3,
                     skip_daylight: bool = False) -> dict:
    """
    Query the JPL Horizons System for a range of times.
    """
    epochs = {'start': starting_datetime, 'stop': ending_datetime,
              'step': step}
    obj = Horizons(id=target, location=location, epochs=epochs)
    if airmass_lessthan is None:
        return obj.ephemerides(skip_daylight=skip_daylight)
    else:
        return obj.ephemerides(airmass_lessthan=airmass_lessthan,
                               skip_daylight=skip_daylight)


def _get_eclipse_indices(ephemeris: dict) -> np.ndarray:
    """
    Search through an ephemeris table and find when a satellite is eclipsed by
    Jupiter and it's either night or astronomical twilight on Maunakea.
    """
    return np.where((ephemeris['sat_vis'] == 'u') &
                    (ephemeris['solar_presence'] != 'C') &
                    (ephemeris['solar_presence'] != 'N') &
                    (ephemeris['solar_presence'] != '*'))[0]
