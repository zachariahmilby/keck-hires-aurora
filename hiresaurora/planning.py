import datetime
from pathlib import Path

import astropy.units as u
import matplotlib.dates as dates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import pytz
from astroplan import Observer
from astropy.coordinates import SkyCoord
from astropy.time import Time
from numpy.typing import ArrayLike

from hiresaurora.ephemeris import _get_ephemerides, _get_eclipse_indices
from hiresaurora.general import color_dict, rcparams

target_info = {'Io': {'ID': '501', 'color': color_dict['red']},
               'Europa': {'ID': '502', 'color': color_dict['blue']},
               'Ganymede': {'ID': '503', 'color': color_dict['grey']},
               'Callisto': {'ID': '504', 'color': color_dict['violet']},
               'Jupiter': {'ID': '599', 'color': color_dict['orange']}}


def _convert_string_to_datetime(time_string: str) -> datetime.datetime:
    """
    Convert an ephemeris table datetime string to a Python datetime object.
    """
    return datetime.datetime.strptime(time_string, '%Y-%b-%d %H:%M')


def _convert_datetime_to_string(datetime_object: datetime.datetime) -> str:
    """
    Convert a Python datetime object to a string with the format
    YYYY-MM-DD HH:MM.
    """
    return datetime.datetime.strftime(datetime_object,
                                      '%Y-%b-%d %H:%M %Z').strip()


def _convert_ephemeris_date_to_string(ephemeris_datetime: str) -> str:
    """
    Ensures an ephemeris datetime is in the proper format.
    """
    return _convert_datetime_to_string(
        _convert_string_to_datetime(ephemeris_datetime))


def _convert_to_california_time(utc_time_string: str) -> datetime.datetime:
    """
    Convert a UTC datetime string to local time at Caltech.
    """
    datetime_object = _convert_string_to_datetime(utc_time_string)
    timezone = pytz.timezone('America/Los_Angeles')
    datetime_object = pytz.utc.localize(datetime_object)
    return datetime_object.astimezone(timezone)


def _calculate_duration(starting_time: str, ending_time: str) -> str:
    """
    Determine duration between two datetime strings to minute precision.
    """
    duration = _convert_string_to_datetime(
        ending_time) - _convert_string_to_datetime(starting_time)
    minutes, seconds = divmod(duration.seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:0>2}'


def _keck_one_alt_az_axis(axis: plt.Axes) -> plt.Axes:
    """
    Modify a default polar axis to be set up for altitude-azimuth plotting.
    Be careful! The input axis must be a polar projection already!
    """
    axis.set_theta_zero_location('N')
    axis.set_theta_direction(-1)  # set angle direction to clockwise
    lower_limit_az = np.arange(np.radians(5.3), np.radians(146.3),
                               np.radians(0.1))
    upper_limit_az = np.concatenate((np.arange(np.radians(146.3),
                                               np.radians(360.0),
                                               np.radians(0.1)),
                                     np.arange(np.radians(0.0),
                                               np.radians(5.4),
                                               np.radians(0.1))))
    lower_limit_alt = np.ones_like(lower_limit_az) * 33.3
    upper_limit_alt = np.ones_like(upper_limit_az) * 18
    azimuth_limit = np.concatenate((lower_limit_az, upper_limit_az,
                                    [lower_limit_az[0]]))
    altitude_limit = np.concatenate((lower_limit_alt, upper_limit_alt,
                                     [lower_limit_alt[0]]))
    axis.fill_between(azimuth_limit, altitude_limit, 0, color='k',
                      alpha=0.5, linewidth=0, zorder=2)
    axis.set_rmin(0)
    axis.set_rmax(90)
    axis.set_yticklabels([])
    axis.set_xticks(np.arange(0, 2 * np.pi, np.pi / 6))
    axis.xaxis.set_tick_params(pad=-3)
    axis.yaxis.set_major_locator(ticker.MultipleLocator(15))
    axis.yaxis.set_minor_locator(ticker.NullLocator())
    axis.set_xticklabels(
        ['N', '', '', 'E', '', '', 'S', '', '', 'W', '', ''])
    axis.grid(linewidth=0.5, zorder=1)
    axis.set_xlabel('Keck I Pointing Limits', fontweight='bold')
    return axis


def _format_axis_date_labels(utc_axis: plt.Axes) -> plt.Axes:
    """
    Format axis date labels so that major ticks occur every hour and minor
    ticks occur every 15 minutes. Also creates a new axis with local
    California time as the upper horizontal axis.
    """
    utc_axis.xaxis.set_major_formatter(dates.DateFormatter('%H:%M'))
    utc_axis.xaxis.set_major_locator(dates.HourLocator(interval=1))
    utc_axis.xaxis.set_minor_locator(
        dates.MinuteLocator(byminute=np.arange(0, 60, 15), interval=1))
    pacific_axis = utc_axis.twiny()
    pacific_axis.set_xlim(utc_axis.get_xlim())
    pacific_axis.xaxis.set_major_formatter(
        dates.DateFormatter('%H:%M', tz=pytz.timezone('US/Pacific')))
    pacific_axis.xaxis.set_major_locator(dates.HourLocator(interval=1))
    pacific_axis.xaxis.set_minor_locator(
        dates.MinuteLocator(byminute=np.arange(0, 60, 15), interval=1))
    return pacific_axis


class _AngularSeparation:
    """
    This class takes a pre-calculated ephemeris table, generates the
    corresponding ephemeris table for a second target, then calculates the
    angular separation between the two targets over the duration of the table.
    """

    def __init__(self, target1_ephemeris: dict, target2_name: str):
        """
        Parameters
        ----------
        target1_ephemeris : dict
            Ephemeris table for the primary object.
        target2_name : str
            The comparison body (Io, Europa, Ganymede, Callisto or Jupiter).
        """
        self._target1_ephemeris = target1_ephemeris
        self._target2_name = target2_name
        self._target2_ephemeris = None
        self._angular_separation = self._calculate_angular_separation()

    def _calculate_angular_separation(self) -> u.Quantity:
        """
        Calculate the angular separation between targets.
        """
        target2_ephemeris = _get_ephemerides(
            target=self._target2_name,
            starting_datetime=self._target1_ephemeris['datetime_str'][0],
            ending_datetime=self._target1_ephemeris['datetime_str'][-1],
            airmass_lessthan=3)
        self._target2_ephemeris = target2_ephemeris
        target_1_positions = SkyCoord(ra=self._target1_ephemeris['RA'],
                                      dec=self._target1_ephemeris['DEC'])
        target_2_positions = SkyCoord(ra=target2_ephemeris['RA'],
                                      dec=target2_ephemeris['DEC'])
        return target_1_positions.separation(target_2_positions).to(u.arcsec)

    @property
    def target_2_ephemeris(self) -> dict:
        return self._target2_ephemeris

    @property
    def values(self) -> u.Quantity:
        return self._angular_separation

    @property
    def angular_radius(self) -> u.Quantity:
        return self._target2_ephemeris['ang_width'].value/2 * u.arcsec


class TelescopePointing:
    """
    Calculate the offsets for manual pointing of the Keck telescope.
    """
    def __init__(self, eclipse_data: dict, guide_satellite: str):
        self._target_satellite_ephemeris = eclipse_data
        self._guide_satellite_ephemeris = \
            _get_ephemerides(target=guide_satellite,
                             starting_datetime=eclipse_data['datetime_str'][0],
                             ending_datetime=eclipse_data['datetime_str'][-1],
                             airmass_lessthan=None)
        self._offsets_guide_to_target = self._calculate_offsets(
            origin=self._guide_satellite_ephemeris,
            destination=self._target_satellite_ephemeris)
        self._offsets_target_to_guide = self._calculate_offsets(
            origin=self._target_satellite_ephemeris,
            destination=self._guide_satellite_ephemeris)

    @staticmethod
    def _calculate_offsets(origin: dict, destination: dict):
        """
        Calculate the offsets for particular starting and ending target
        ephemeris tables.
        """
        origin_ra = origin['RA_app']
        destination_ra = destination['RA_app']
        ra_offset = ((destination_ra - origin_ra) * u.degree).to(u.arcsec)
        dra = destination['RA_rate'].to(u.arcsec / u.s) / 15

        origin_dec = origin['DEC_app']
        destination_dec = destination['DEC_app']
        dec_offset = ((destination_dec - origin_dec) * u.degree).to(u.arcsec)
        ddec = destination['DEC_rate'].to(u.arcsec / u.s)

        npang = destination['NPole_ang']

        return {
            'name': destination['targetname'][0],
            'time': destination['datetime_str'],
            'ra': destination_ra,
            'ra_offset': ra_offset,
            'dra': dra,
            'dec': destination_dec,
            'dec_offset': dec_offset,
            'ddec': ddec,
            'npang': npang,
            'nlines': len(origin_ra),
        }

    @property
    def offsets_guide_to_target(self):
        return self._offsets_guide_to_target

    @property
    def offsets_target_to_guide(self):
        return self._offsets_target_to_guide


class EclipsePrediction:
    """
    This class finds all instances of a target (Io, Europa, Ganymede or
    Callisto) being eclipsed by Jupiter over a given timeframe visible from
    Mauna Kea at night.

    Parameters
    ----------
    starting_datetime : str
        The date you want to begin the search. Format (time optional):
        YYYY-MM-DD [HH:MM:SS].
    ending_datetime
        The date you want to end the search. Format (time optional):
        YYYY-MM-DD [HH:MM:SS].
    target : str
        The Galilean moon for which you want to find eclipses. Io, Europa,
        Ganymede or Callisto.
    """

    def __init__(self, starting_datetime: str, ending_datetime: str,
                 target: str):
        self._target = target_info[target]['ID']
        self._target_name = target
        self._starting_datetime = starting_datetime
        self._ending_datetime = ending_datetime
        self._eclipses = self._find_eclipses()
        print(self._print_string())

    def __str__(self):
        return self._print_string()

    @staticmethod
    def _consecutive_integers(
            data: np.ndarray, stepsize: int = 1) -> np.ndarray:
        """
        Find sets of consecutive integers (find independent events in an
        ephemeris table).
        """
        return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)

    def _find_eclipses(self) -> list[dict]:
        """
        Find the eclipses by first querying the JPL Horizons System in 1-hour
        intervals, finding the eclipse events, then performs a more refined
        search around those events in 1-minute intervals.
        """
        data = []
        initial_ephemeris = _get_ephemerides(
            target=self._target, starting_datetime=self._starting_datetime,
            ending_datetime=self._ending_datetime, step='1h')
        eclipses = self._consecutive_integers(
            _get_eclipse_indices(initial_ephemeris))
        if len(eclipses[-1]) == 0:
            raise Exception('Sorry, no eclipses found!')
        for eclipse in eclipses:
            self._target_name = \
                initial_ephemeris['targetname'][0].split(' ')[0]
            starting_time = _convert_string_to_datetime(
                initial_ephemeris[eclipse[0]]['datetime_str'])
            starting_time -= datetime.timedelta(days=1)
            starting_time = _convert_datetime_to_string(starting_time)
            ending_time = _convert_string_to_datetime(
                initial_ephemeris[eclipse[-1]]['datetime_str'])
            ending_time += datetime.timedelta(days=1)
            ending_time = _convert_datetime_to_string(ending_time)
            ephemeris = _get_ephemerides(
                target=self._target, starting_datetime=starting_time,
                ending_datetime=ending_time, step='1m')
            indices = _get_eclipse_indices(ephemeris)
            refined_ephemeris = _get_ephemerides(
                target=self._target,
                starting_datetime=ephemeris['datetime_str'][indices[0]],
                ending_datetime=ephemeris['datetime_str'][indices[-1]])
            data.append(refined_ephemeris)
        return data

    def _print_string(self) -> str:
        """
        Format a terminal-printable summary table of the identified eclipses
        along with starting/ending times in both UTC and local California time,
        the duration of the eclipse, the range in airmass and the satellite's
        relative velocity.
        """
        print(f'\n{len(self._eclipses)} {self._target_name} eclipse(s) '
              f'identified between {self._starting_datetime} and '
              f'{self._ending_datetime}.\n')
        df = pd.DataFrame(
            columns=['Starting Time (Keck/UTC)', 'Ending Time (Keck/UTC)',
                     'Starting Time (California)', 'Ending Time (California)',
                     'Duration', 'Airmass Range', 'Relative Velocity'])
        for eclipse in range(len(self._eclipses)):
            times = self._eclipses[eclipse]['datetime_str']
            airmass = self._eclipses[eclipse]['airmass']
            relative_velocity = np.mean(self._eclipses[eclipse]['delta_rate'])
            starting_time_utc = times[0]
            ending_time_utc = times[-1]
            data = {
                'Starting Time (Keck/UTC)':
                    _convert_ephemeris_date_to_string(starting_time_utc),
                'Ending Time (Keck/UTC)':
                    _convert_ephemeris_date_to_string(ending_time_utc),
                'Starting Time (California)': _convert_datetime_to_string(
                    _convert_to_california_time(starting_time_utc)),
                'Ending Time (California)': _convert_datetime_to_string(
                    _convert_to_california_time(ending_time_utc)),
                'Duration':
                    _calculate_duration(starting_time_utc, ending_time_utc),
                'Airmass Range':
                    f"{np.min(airmass):.3f} to {np.max(airmass):.3f}",
                'Relative Velocity': f"{relative_velocity:.3f} km/s"
            }
            df = pd.concat([df, pd.DataFrame(data, index=[0])])
        return pd.DataFrame(df).to_string(index=False, justify='left')

    @staticmethod
    def _plot_line_with_initial_position(
            axis: plt.Axes, x: ArrayLike, y: u.Quantity, color: str,
            label: str = None, radius: u.Quantity = None) -> None:
        """
        Plot a line with a scatterplot point at the starting position. Useful
        so I know on different plots which point corresponds to the beginning
        of the eclipse.

        Update 2022-05-11: now includes Jupiter's angular diameter.
        """
        axis.plot(x, y, color=color, linewidth=1)
        if radius is not None:
            axis.fill_between(x, y.value+radius.value, y.value-radius.value,
                              color=color, linewidth=0, alpha=0.25)
        axis.scatter(x[0], y[0], color=color, edgecolors='none', s=9)
        if label is not None:
            axis.annotate(label, xy=(x[0], y[0].value), va='center',
                          ha='right', xytext=(-3, 0), fontsize=6,
                          textcoords='offset pixels', color=color)

    def save_summary_graphics(self, save_directory: str = Path.cwd()) -> None:
        """
        Save a summary graphic of each identified eclipse to a specified
        directory.
        """
        for eclipse in range(len(self._eclipses)):

            # get relevant quantities
            times = self._eclipses[eclipse]['datetime_str']
            starting_time = times[0]
            ending_time = times[-1]
            duration = _calculate_duration(starting_time, ending_time)
            times = dates.datestr2num(times)
            polar_angle = 'unknown'
            observer = Observer.at_site('Keck')
            sunset = observer.sun_set_time(
                Time(_convert_string_to_datetime(starting_time)),
                which='nearest')
            sunset = _convert_datetime_to_string(sunset.datetime)
            sunrise = observer.sun_rise_time(
                Time(_convert_string_to_datetime(ending_time)),
                which='nearest')
            sunrise = _convert_datetime_to_string(sunrise.datetime)

            # make figure and place axes
            with plt.style.context(rcparams):
                fig = plt.figure(figsize=(5, 4), constrained_layout=True)
                gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1.5],
                                       figure=fig)
                info_axis = fig.add_subplot(gs[0, 0])
                info_axis.set_frame_on(False)
                info_axis.set_xticks([])
                info_axis.set_yticks([])
                alt_az_polar_axis = _keck_one_alt_az_axis(
                    fig.add_subplot(gs[1, 0], projection='polar'))
                airmass_axis_utc = fig.add_subplot(gs[0, 1])
                airmass_axis_utc.set_ylabel('Airmass', fontweight='bold')
                primary_sep_axis_utc = fig.add_subplot(gs[1, 1])
                primary_sep_axis_utc.set_ylabel('Angular Separation [arcsec]',
                                                fontweight='bold')

                # plot data
                self._plot_line_with_initial_position(
                    alt_az_polar_axis,
                    np.radians(self._eclipses[eclipse]['AZ']),
                    self._eclipses[eclipse]['EL'], color='k')
                self._plot_line_with_initial_position(
                    airmass_axis_utc, times,
                    self._eclipses[eclipse]['airmass'],
                    color='k')
                airmass_axis_california = _format_axis_date_labels(
                    airmass_axis_utc)
                for ind, target in enumerate(
                        [target_info[key]['ID']
                         for key in target_info.keys()]):
                    angular_separation = _AngularSeparation(
                        self._eclipses[eclipse], target)
                    # get Jupiter's average polar angle rotation when
                    # calculating it's ephemerides
                    radius = 0 * u.arcsec
                    if target == '599':
                        polar_angle = np.mean(
                            angular_separation.target_2_ephemeris['NPole_ang'])
                        radius = angular_separation.angular_radius
                    if np.sum(angular_separation.values != 0):
                        self._plot_line_with_initial_position(
                            primary_sep_axis_utc, times,
                            angular_separation.values,
                            color=target_info[
                                list(target_info.keys())[ind]]['color'],
                            label=angular_separation.target_2_ephemeris[
                                'targetname'][0][0], radius=radius)
                primary_sep_axis_california = _format_axis_date_labels(
                    primary_sep_axis_utc)

                # information string, it's beastly but I don't know a better
                # way of doing it...
                info_string = 'California Start:' + '\n'
                info_string += 'California End:' + '\n' * 2
                info_string += 'Keck Start:' + '\n'
                info_string += 'Keck End:' + '\n' * 2
                info_string += 'Keck Sunset:' + '\n'
                info_string += 'Keck Sunrise:' + '\n' * 2
                info_string += f'Duration:   {duration}' + '\n'
                info_string += 'Jupiter North Pole Angle:   '
                info_string += fr"{polar_angle:.1f}$\degree$" + '\n'
                info_string += f'{self._target_name} Relative Velocity:   '
                info_string += \
                    fr"${np.mean(self._eclipses[eclipse]['delta_rate']):.3f}$ "
                info_string += 'km/s'
                times_string = _convert_datetime_to_string(
                    _convert_to_california_time(starting_time))
                times_string += '\n'
                times_string += _convert_datetime_to_string(
                    _convert_to_california_time(ending_time))
                times_string += '\n' * 2
                times_string += f'{starting_time} UTC' + '\n'
                times_string += f'{ending_time} UTC' + '\n'
                times_string += '\n'
                times_string += f'{sunset} UTC' + '\n'
                times_string += f'{sunrise} UTC'
                info_axis.text(0.05, 0.95, info_string, linespacing=1.67,
                               ha='left', va='top', fontsize=6)
                info_axis.text(0.4, 0.95, times_string, linespacing=1.67,
                               ha='left', va='top',
                               transform=info_axis.transAxes, fontsize=6)
                info_axis.set_title('Eclipse Information', fontweight='bold')

                # set axis labels, limits and other parameters
                airmass_axis_california.set_xlabel('Time (California)',
                                                   fontweight='bold')
                airmass_axis_utc.set_xticklabels([])
                primary_sep_axis_utc.set_xlabel('Time (UTC)',
                                                fontweight='bold')
                primary_sep_axis_california.set_xticklabels([])
                alt_az_polar_axis.set_rmin(90)
                alt_az_polar_axis.set_rmax(0)
                airmass_axis_utc.set_ylim(1, 2)
                primary_sep_axis_utc.set_ylim(bottom=0)

                # save the figure
                filename_date_str = datetime.datetime.strftime(
                    _convert_string_to_datetime(starting_time), '%Y-%m-%d')
                filepath = Path(save_directory,
                                f'{self._target_name.lower()}_'
                                f'{filename_date_str.lower()}.pdf')
                if not filepath.parent.exists():
                    filepath.mkdir(parents=True)
                plt.savefig(filepath)
                plt.close(fig)

    @staticmethod
    def offset_position(ind: int, offsets: dict) -> str:
        return f"{offsets['name'].split(' ')[0]} " \
               f"{offsets['time'][ind].split(' ')[1]}   " \
               f"en {offsets['ra_offset'][ind].value:.3f} " \
               f"{offsets['dec_offset'][ind].value:.3f}\n"

    @staticmethod
    def offset_rates(ind: int, offsets: dict) -> str:
        return f"{offsets['name'].split(' ')[0]} " \
               f"{offsets['time'][ind].split(' ')[1]}   " \
               f"modify -s dcs dtrack=1 dra={offsets['dra'][ind].value:.9f} " \
               f"ddec={offsets['ddec'][ind].value:.9f}\n"

    def save_pointing_offsets(self, guide_satellite,
                              save_directory: str = Path.cwd()):
        """
        Save calculated pointing offsets and rates.

        Parameters
        ----------
        guide_satellite : str
            The name of the guide satellite.
        save_directory: str or Path
            The location where you want the two text files saved.

        Returns
        -------
        None.
        """
        if isinstance(save_directory, Path):
            save_directory = str(Path)
        for eclipse in self._eclipses:
            pointing_information = TelescopePointing(
                eclipse, guide_satellite=target_info[guide_satellite]['ID'])
            date = eclipse['datetime_str'][0].split(' ')[0]

            # guide to target
            info = pointing_information.offsets_guide_to_target
            guide_to_target_offsets = Path(
                save_directory,
                f'offsets_{guide_satellite}_to_{self._target_name}_{date}.txt')
            if not guide_to_target_offsets.parent.exists():
                guide_to_target_offsets.parent.mkdir(parents=True)
            with open(guide_to_target_offsets, 'w') as file:
                for i in range(info['nlines']):
                    file.write(self.offset_position(i, info))
            guide_to_target_rates = Path(
                save_directory,
                f'rates_{guide_satellite}_to_{self._target_name}_{date}.txt')
            if not guide_to_target_rates.parent.exists():
                guide_to_target_rates.parent.mkdir(parents=True)
            with open(guide_to_target_rates, 'w') as file:
                for i in range(info['nlines']):
                    file.write(self.offset_rates(i, info))

            # target to guide
            info = pointing_information.offsets_target_to_guide
            target_to_guide_offsets = Path(
                save_directory,
                f'offsets_{self._target_name}_to_{guide_satellite}_{date}.txt')
            if not target_to_guide_offsets.parent.exists():
                target_to_guide_offsets.parent.mkdir(parents=True)
            with open(target_to_guide_offsets, 'w') as file:
                for i in range(info['nlines']):
                    file.write(self.offset_position(i, info))
            target_to_guide_rates = Path(
                save_directory,
                f'rates_{self._target_name}_to_{guide_satellite}_{date}.txt')
            if not target_to_guide_rates.parent.exists():
                target_to_guide_rates.parent.mkdir(parents=True)
            with open(target_to_guide_rates, 'w') as file:
                for i in range(info['nlines']):
                    file.write(self.offset_rates(i, info))
