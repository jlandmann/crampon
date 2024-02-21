import numpy as np
import xarray as xr
import pandas as pd
from crampon import utils
import datetime as dt
import glob
import os
from crampon import cfg

all_stations = [1001, 1002, 1003, 1006, 1007, 1008, 1009]

id_to_station = {
    'RGI50-11.B4312n-1': [1002, 1006, 1007, 1009],
    'RGI50-11.A55F03': [1003],
    'RGI50-11.B5616n-1': [1001, 1008]
}

# todo: put all data into one single xr Dataset with time axis
station_coordinates = {

}

station_to_glacier = {
    1001: 'RGI50-11.B5616n-1',
    1002: 'RGI50-11.B4312n-1',
    1003: 'RGI50-11.A55F03',
    1006: 'RGI50-11.B4312n-1',
    1007: 'RGI50-11.B4312n-1',
    1008: 'RGI50-11.B5616n-1',
    1009: 'RGI50-11.B4312n-1',
}

station_to_height = {
    1001: 3021,  # ,3026
    1002: 2235,  # ,2255
    1003: 2681,  # ,2689
    1006: 2392,  # ,2403
    1007: 2589,  # ,2596
    1008: 2564,  # 2584
    1009: 2233,  # 2252
}


class HolfuyStation(object):
    """ Interface to the Holfuy camera station."""

    def __init__(self, number, img_path=None, mread_path=None):

        self.number = number
        if img_path is not None:
            self.img_path = img_path
        else:
            self.img_path = None

        if mread_path is not None:
            self.mread_path = mread_path
        else:
            self.mread_path = None

        self.acquisition_interval = cfg.SEC_IN_HOUR * 20  # 20min

        if self.number in [1001, 1002, 1003]:
            self.has_weather_data = True
        else:
            self.has_weather_data = False

        self._is_operational = None
        self._images = None
        self._image_dates = None

    @property
    def images(self):
        """
        Images that  belong to one station.
        """
        return self._images

    @images.getter
    def images(self):
        """Images getter"""
        if self.img_path is not None:
            return glob.glob(os.path.join(self.img_path, '.png'))
        else:
            raise ValueError('img_path must be defined to get station images.')

    @property
    def image_dates(self):
        """Dates for all images."""
        return self._image_dates

    @image_dates.getter
    def image_dates(self):
        """Image date getter."""
        dates = [dt.datetime.strptime(x.split('.')[0], '%Y-%m-%d_%H-%M')
                 for x in self.images]
        return dates

    @property
    def is_operational(self):
        """Whether or not the station is operational."""
        return self._is_operational

    @is_operational.getter
    def is_operational(self):
        """Getter for operational property."""
        now = dt.datetime.now()

        if (now - max(self.image_dates)).total_seconds() <= \
                self.acquisition_interval:
            return True
        else:
            return False

    @utils.lazy_property
    def manual_readings(self, **kwargs):
        """
        Manual readings at the station.

        Parameters
        ----------
        kwargs : dict
            Parameters passed on to `utils.read_holfuy_camera_reading`.

        Returns
        -------
        mr: pd.Dataframe
            Dataframe containing the manual readings.
        """
        mr = utils.read_holfuy_camera_reading(self.mread_path, **kwargs)
        mr['swe'] = utils.obs_operator_dh_stake_to_mwe(mr['dh'])
        mr['swe_uncertainty'] = \
            utils.obs_operator_dh_stake_to_mwe(mr['uncertainty'])
        return mr

    def make_animation(self, from_datetime=None, to_datetime=None):
        """
        Make an animation with all or parts of the images from this station.

        Parameters
        ----------
        from_datetime
        to_datetime

        Returns
        -------

        """
        raise NotImplementedError
        # holfuyretriever.visualization.make_image_animation(path,
        # save_path=None, speedup='auto')


def read_holfuy_camera_reading(fpath, ice_only=True, exclude_keywords=True):
    """
    Read a CSV file with Holfuy camera readings.

    Parameters
    ----------
    fpath: str
        Path to the CSV file.
    ice_only: bool
        If True, only days where the phase is ice are left. Default: True.
    exclude_keywords: bool
        If True, days that should be excluded, i.e. when the station was
        redrilled, set up, or torn down, are cut out from the data. Default:
        True.

    Returns
    -------
    cread: pd.DataFrame
        Pandas DataFrame with camera readings.
    """
    cread = pd.read_csv(fpath, index_col=0, parse_dates=[0])

    # exclude some of the critical days
    if exclude_keywords:
        cread = cread[~cread.key_remarks.str.contains("SETUP", na=False)]
        # cread = cread[~cread.key_remarks.str.contains("REDRILL", na=False)]
        cread = cread[~cread.key_remarks.str.contains("TEARDOWN", na=False)]

    if ice_only:
        cread = cread[cread.phase == 'i']  # exclude snow

    return cread


def obs_operator_dh_stake_to_mwe(dh_observation, rho=None):
    """
    Observation operator turning dh stake readings into meter water equivalent.

    Parameters
    ----------
    dh_observation: array-like, float
        Surface hieght change reading from a mass balance stake or camera.
        for a surface lowering, dh_observation must be POSITIVE.
    rho: float or None
        Density of the matter (snow, slush or ice, for example). Default is
        None and takes the density of ice from the configuration.

    Returns
    -------
    mwe: array-like, float
        Height change converted to meter water equivalent
    """

    if rho is None:
        rho = cfg.RHO

    mwe = np.negative(dh_observation) * rho / cfg.RHO_W
    return mwe


def prepare_holfuy_camera_readings(gdir, ice_only=True,
                                   exclude_initial_snow=True,
                                   exclude_keywords=True,
                                   rho_fresh_snow_guess=None,
                                   stations=None,
                                   holfuy_path=''):
    """
    Read stations per glacier, drop snow & convert dh into water equivalent.

    Parameters
    ----------
    gdir: utils.GlacierDirectory
        The GlacierDirectory to process the readings for.
    ice_only: bool
        If True, only days where the phase is ice are left. Default: True.
    exclude_initial_snow: bool
        An additional keyword that excludes the first phase when there was
        still snow snow and the camera was set up. This is thought for the
        case when we include snow generally (ice_only=False), but want to
        omit the uncertainty that snow causes in the beginning. Default:True.
    exclude_keywords: bool
        If True, days that should be excluded, i.e. when the station was
        redrilled, set up, or torn down, are cut out from the data. Default:
        True.
    rho_fresh_snow_guess: float
        A guess for the density of resh snow. Default: 150 kg m-3.
    stations: list
        List of station numbers (int) to use. If None, use all stations per
        gdir. Default: None.
    holfuy_path: str
        # todo: having this an empty string by default is bullshit. It should
           be a mandatory parameter.
        Path to where the Holfuy readings are.

    Returns
    -------
    obs_merge: pd.DataFrame
        Dataframe with the measurement converted to water equivalent.
    """
    if stations is None:
        stations = id_to_station[gdir.rgi_id]
    else:
        stations = stations

    if rho_fresh_snow_guess is None:
        rho_fresh_snow_guess = cfg.PARAMS['rho_fresh_snow']

    # todo: make path flexible
    meas_list = [read_holfuy_camera_reading(
        holfuy_path + 'manual_reading_{}.csv'.format(
            s), ice_only=ice_only, exclude_keywords=exclude_keywords) for s in
        stations]
    for i in range(len(meas_list)):
        # meas_list[i].loc[meas_list[i].phase == 'i', 'swe'] =
        # obs_operator_dh_stake_to_mwe(meas_list[i].loc[meas_list[i].phase
        # == 'i','dh'])
        # meas_list[i].loc[meas_list[i].phase == 'i', 'swe_std'] =
        # - obs_operator_dh_stake_to_mwe(meas_list[i].loc[meas_list[i].phase
        # == 'i', 'uncertainty'])

        # total SWE = SWE_ICE_MELT + SWE_FRESH_SNOW
        meas_list[i]['swe'] = \
            obs_operator_dh_stake_to_mwe(meas_list[i]['dh'])
        meas_list[i]['swe_std'] = \
            - obs_operator_dh_stake_to_mwe(meas_list[i]['uncertainty'])

        # fill all the not given values with the standard value
        meas_list[i]['swe_std'] = \
            meas_list[i]['swe_std'].fillna(-utils.obs_operator_dh_stake_to_mwe(
                cfg.PARAMS['dh_obs_manual_std_default']))

        if 'fresh_snow_height' in meas_list[i].columns:
            meas_list[i]['fsnow_swe'] = meas_list[i]['fresh_snow_height'] * \
                                        rho_fresh_snow_guess / 1000.
            meas_list[i]['fsnow_swe'] = meas_list[i]['fsnow_swe'].fillna(0.)
            meas_list[i]['fsnow_swe_std'] = \
                meas_list[i]['fresh_snow_height_uncertainty'] * \
                rho_fresh_snow_guess / 1000.

            # add it to total SWE (skip NaNs automatically)
            meas_list[i]["swe"] = meas_list[i][["swe", "fsnow_swe"]].sum(
                axis=1, skipna=False)
            # add errors independently
            # todo: check if adding errors is the correct way to do it
            meas_list[i]['swe_std'] = np.sqrt(meas_list[i][
                                                  'swe_std'].fillna(
                0.).values**2, meas_list[i]['fsnow_swe_std'].fillna(
                0.).values**2)

        # todo: convert snow hieght change to swe here!?
        meas_list[i].drop('dh', axis=1, inplace=True)

        if exclude_initial_snow:
            first_ice_ix = meas_list[i].phase.eq('i').idxmax()
            meas_list[i] = meas_list[i].loc[first_ice_ix:]

    obs_list = [m.to_xarray() for m in meas_list]
    obs_merge = xr.concat(obs_list, pd.Index(
        [station_to_height[s] for s in stations], name='height'))

    return obs_merge
