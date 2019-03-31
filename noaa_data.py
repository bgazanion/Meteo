import pandas as pd
import os
import re
import matplotlib.pyplot as plt


def get_duplicates_of(df, name):
    """
    Return the names of columns that correspond to duplicates: X, X -> X, X.1

    :param df: pandas dataframe
    :param name: original name (eg. X for X.1)
    :return: list of strings
    """
    regex = re.compile(r'^(__name__(?:\.\d+)?)$'.replace('__name__', name))
    return [c for c in df.columns if bool(regex.match(c))]


def read_station(station_file):
    """
    Read the station name corresponding to the USAF code in the station file

    :param station_file:
    :return: dict
    """
    if os.path.exists(station_file):
        with open(station_file, 'r') as f_station:
            content = f_station.readlines()

        # content: USAF, WBAN, STATION_NAME, COUNTRY, ...
        header = content[0]
        station_start = header.find('STATION NAME')
        country_start = header.find('COUNTRY')

        # read content (note: skip delimiter line)
        stations = {}
        for line in content[2:] :
            usaf = line.split()[0].strip()
            name = line[station_start: country_start-1].strip()
            stations[usaf] = name

        return stations

    else:
        # could not find the station file
        return None


def read_data(data_file, station_file=None, light_output=True):
    """
    Read a meteo dataset

    :param data_file:
    :param station_file:
    :return: pandas dataframe
    """

    # read file
    df = pd.read_csv(data_file,
                     delim_whitespace=True,
                     na_values=['*', '**', '***', '****', '*****'],
                     dtype={'YR--MODAHRMN': str, 'USAF': str, 'SKC': str})

    # use date as index
    date = pd.to_datetime(df['YR--MODAHRMN'],
                             format='%Y%m%d%H%M')
    df.pop('YR--MODAHRMN')
    df.set_index(date, inplace=True)

    # direction: NaN values are given as 990
    df.loc[df['DIR'] > 370, 'DIR'] = pd.np.NaN

    # station : station names if station_file is provided, USAF code otherwise
    if station_file:
        stations = read_station(station_file)
        if stations:
            # util function
            def usaf_to_station(usaf):
                if usaf in stations:
                    return stations[usaf]
                else:
                    return usaf
            # apply util function to modify 'station'
            df['station'] = df['USAF'].apply(usaf_to_station).astype('category')

    return df


def select_data(df, basic=False, cloud=False,
                    precipitation=False, observation=False):
    """
    Return a profile of meteo data based on output options.
    Notice that if all output options are deactivated (False), the original
    dataframe is returned.

    :param df: input pandas dataframe
    :param basic: output contains DIR, SPD, TEMP, DEWP
    :param cloud: output contains CLG, SKC, L, M, H
    :param precipitation: output contains PCP01, PCP06, PCP24, PCPXX
    :param observation: output contains W, and MW & AW with renamed duplic.
    :return: dataframe with selected columns
    """

    if sum([basic, cloud, precipitation, observation]) == 0:
        # default: return original dataframe
        return df
    else:
        # create list of columns in the output
        list_output = ['station', 'WBAN', 'USAF']
        if basic:
            list_output += ['DIR', 'SPD', 'TEMP', 'DEWP']
        if cloud:
            list_output += ['CLG', 'SKC', 'L', 'M', 'H']
        if precipitation:
            list_output += ['PCP01', 'PCP06', 'PCP24', 'PCPXX']
        if observation:
            list_output.append('W')
            list_output += get_duplicates_of(df, 'MW')
            list_output += get_duplicates_of(df, 'AW')

        # clean list: keep only names of actual columns
        list_clean = [s for s in list_output if s in df.columns]

        # create output
        return df[list_clean]


def convert(serie, conversion):
    """
    Convert physical quantities from units to others

    :param serie: input pandas serie
    :param conversion: conversion string
    :return: pandas serie
    """
    if conversion == 'fahrenheit_to_celsius':
        return (5./9.)*(serie-32)
    if conversion == 'fahrenheit_to_kelvin':
        return (5./9.)*(serie+459.67)
    if conversion == 'celsius_to_kelvin':
        return serie+273.15
    if conversion == 'kelvin_to_celsius':
        return serie-273.15
    if conversion == 'inch_to_m':
        return serie*0.0254
    if conversion == 'mile_to_m':
        return serie*1609.344
    if conversion == 'mile/h_to_m/s':
        return serie*0.44704
    if conversion == 'km/h_to_m/s':
        return serie*0.2777778
    if conversion == 'mbar_to_N.m2':
        return serie*100.
    else:
        print('Unknown conversion')
        return serie*0.


def select_station(df, name):
    """
    Extract the data corresponding to a station

    :param df: dataframe
    :param name: name of the station, or USAF code if df has no column 'station'
    :return: sub-dataframe for the station
    """

    # get stations
    if 'station' in df.columns:
        stations = list_stations(df)
        if name in stations:
            return df[df['station'] == name]
        else:
            raise ValueError('Dataframe has no station nammed "'+name+'"')
    else :
        usaf = list_stations(df)
        if name in usaf:
            return df[df['USAF'] == name]
        else:
            raise ValueError('Dataframe has no station with USAF "'+name+'"')


def list_stations(df):
    """
    Return the list of stations, or USAF codes, for the dataframe

    :param df: input dataframe
    :return:
    """

    if 'station' in df.columns:
        return list(df['station'].unique())
    else:
        return list(df['USAF'].unique())


def select_time(df, first_day, last_day=None, years=None, months=None,
                weeks=None, days=None):
    """
    Extract a period of time. Notice that the input dataframe must not contain
    more than one station.

    :param df: input dataframe
    :param first_day: date of the first day in YYYY/MM/DD format
    :param last_day: date of the last day in YYYY/MM/DD format (optional)
    :param years: number of years (optional)
    :param months: number of months (optional)
    :param weeks: number of weeks (optional)
    :param days: number of days (optional)
    :return: dataframe
    """

    date_format = '%Y/%m/%d'
    first_datetime = pd.to_datetime(first_day, format=date_format)

    if last_day:
        # mode 1: time defined by first and last day
        last_datetime = pd.to_datetime(last_day, format=date_format)
        return df[first_datetime:last_datetime]

    else:
        # mode 2: time defined by first day and period
        last_datetime = first_datetime

        # handle errors
        if not any([days, weeks, months, years]):
            error = 'Give at least one of: "days", "weeks", "months" or "years"'
            raise ValueError(error)

        # increment last_datetime in that order: year, month, week, day
        if years:
            for year_index in range(years):
                if last_datetime.year % 4 == 0:
                    last_datetime += pd.to_timedelta(366, unit='d')
                else:
                    last_datetime += pd.to_timedelta(365, unit='d')

        if months:
            for month_index in range(months):
                days_in_month = last_datetime.days_in_month
                last_datetime += pd.to_timedelta(days_in_month, unit='M')

        if weeks:
            last_datetime += pd.to_timedelta(weeks, unit='w')

        if days:
            last_datetime += pd.to_timedelta(days, unit='d')

    # result
    df_sliced = df[first_datetime:last_datetime]
    if not last_day:
        # ensure last_datetime is excluded from output
        if df_sliced.index[-1] == last_datetime:
            df_sliced = df_sliced.iloc[:-1]
    return df_sliced


def convert_to_si(df):
    """
    Convert all fields to SI units

    :param df: input dataframe
    :return: dataframe
    """

    df_out = df.copy()
    columns = df.columns

    if 'SPD' in columns:
        df_out['SPD'] = convert(df_out['SPD'], 'mile/h_to_m/s')
    if 'GUS' in columns:
        df_out['GUS'] = convert(df_out['GUS'], 'mile/h_to_m/s')
    if 'VSB' in columns:
        df_out['VSB'] = convert(df_out['VSB'], 'mile_to_m')
    if 'TEMP' in columns:
        df_out['TEMP'] = convert(df_out['TEMP'], 'fahrenheit_to_celsius')
    if 'DEWP' in columns:
        df_out['DEWP'] = convert(df_out['DEWP'], 'fahrenheit_to_celsius')
    if 'SLP' in columns :
        df_out['SLP'] = convert(df_out['SLP'], 'mbar_to_N.m2')
    if 'ALT' in columns :
        df_out['ALT'] = convert(df_out['ALT'], 'inch_to_m')
    if 'STP' in columns :
        df_out['STP'] = convert(df_out['STP'], 'mbar_to_N.m2')
    if 'MAX' in columns:
        df_out['MAX'] = convert(df_out['MAX'], 'fahrenheit_to_celsius')
    if 'MIN' in columns:
        df_out['MIN'] = convert(df_out['MIN'], 'fahrenheit_to_celsius')
    if 'PCP01' in columns:
        df_out['PCP01'] = convert(df_out['PCP01'], 'inch_to_m')
    if 'PCP06' in columns:
        df_out['PCP06'] = convert(df_out['PCP06'], 'inch_to_m')
    if 'PCP24' in columns:
        df_out['PCP24'] = convert(df_out['PCP24'], 'inch_to_m')
    if 'PCPXX' in columns:
        df_out['PCPXX'] = convert(df_out['PCPXX'], 'inch_to_m')
    if 'SD' in columns:
        df_out['SD'] = convert(df_out['SD'], 'inch_to_m')

    return df_out


def polar_stat(df, column, values=[], bounds=False):
    """
    Compute dispersion of 'column' between wind direction and range of values

    :param df: input dataframe
    :param column: target of the statistics
    :param values: values defining the intervals of analysis
    :param bounds: add the bounds, ie intervals with min and max (boolean)
    :return: dataframe of percentage for each direction and range of values
    """

    # handle error
    if column not in df.columns:
        print('Column '+column+' not found in dataframe')
        return None

    # number of valid values in the direction column (DIR)
    nb_valid = df['DIR'].notnull().sum()

    # create output dataframe
    df_polar = pd.DataFrame(index=df['DIR'].unique().sort())
    df_polar.index.name = 'DIR'

    if values == []:
        # compute stats for all values
        series = df.groupby('DIR')[column].count()/nb_valid
        df_polar[column] = series*100

    else:
        # compute stats for lower bound: < first value
        if bounds:
            name = column + ' < ' + str(values[0])
            selection = (df[column] < values[0])
            series = df.loc[selection].groupby('DIR')[column].count()/nb_valid
            df_polar[name] = series*100

        for i in range(len(values)-1):
            # compute stats for range [values[i], values[i+1][
            name = str(values[i]) + ' <= ' + column + ' < ' + str(values[i+1])
            selection = (df[column] >= values[i]) & (df[column] < values[i+1])
            series = df.loc[selection].groupby('DIR')[column].count()/nb_valid
            df_polar[name] = series*100

        # compute stats for upper bound : > last value
        if bounds:
            name = str(values[-1]) + ' <= ' + column
            selection = (df[column] >= values[-1])
            series = df.loc[selection].groupby('DIR')[column].count()/nb_valid
            df_polar[name] = series*100

    return df_polar


def polar_plot(df_polar, close=False, output=None):
    """
    Create a polar plot. The plot is saved as a figure if the name of an output
    file is given ("output"), or displayed.

    :param df_polar: input dataframe from the polar_stats function
    :param close: close the polar plot (boolean)
    :param output: name of the output
    :return:
    """

    # check input
    if len(df_polar.columns) == 0:
        raise ValueError('Empty polar dataframe')

    if close:
        # add a line to a copy, to close the directions
        df_plot = pd.concat([ df_polar, df_polar.iloc[[0]] ])
    else:
        df_plot = df_polar

    # directions in radians (for polar function)
    direction_rad = pd.np.radians(df_plot.index)

    # create figure
    figure = plt.figure()
    axes = figure.add_subplot(111, polar=True)
    axes.set_theta_zero_location('N')
    axes.set_theta_direction('clockwise')
    figure.subplots_adjust(left=0.08, right=0.6)

    # add curves
    for column in df_plot.columns:
        axes.plot(direction_rad, df_plot[column], label=column, linewidth=2)

    # finalize
    legend = plt.legend(bbox_to_anchor=(1.1, 1), loc=2)

    # display or save figure
    if output and type(output) == str:
        plt.savefig(output, bbox_extra_artists=[legend], bbox_inches='tight')
    else:
        figure.set_size_inches(10, 6, forward=True)
        plt.show()