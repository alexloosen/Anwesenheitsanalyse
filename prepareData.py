import numpy as np
import pandas as pd

def encodeCyclical(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def detectOutliers(df):
    x = df['co2_ppm']
    q1 = np.percentile(x, 12)
    q3 = np.percentile(x, 88)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
    outlier_values = list(x[outlier_indices])
    print(outlier_values)
    return outlier_indices

def extractSection(df, dateFrom, dateTo):
    df_new = df.loc[(df['timestamp'] > pd.to_datetime(dateFrom, unit='s', origin='unix')) &
                    (df['timestamp'] < pd.to_datetime(dateTo, unit='s', origin='unix'))]
    return df_new

def dropSection(df, dateFrom, dateTo):
    df_new = df.drop(df[(df['timestamp'] > pd.to_datetime(dateFrom,unit='s',origin='unix')) &
                        (df['timestamp'] < pd.to_datetime(dateTo,unit='s',origin='unix'))].index)
    return df_new

def convertTimestamp(df):
    # timestamp etwas leichter zu verarbeiten, wenn als Integer gespeichert
    df = df.assign(hour=lambda d: (d['timestamp'].dt.hour.astype('int') * 10000 + 
                                   d['timestamp'].dt.minute.astype('int') * 100 + 
                                   d['timestamp'].dt.second.astype('int')))
    return df

def dropBetweenTimestamp(df, timeFrom, timeTo):
    df_new = df.drop(df[(df['hour'] > timeFrom) |
                        (df['hour'] < timeTo)].index, inplace=False)
    return df_new

def addDelta(df, col, amount):
    df[col + '_delta' + str(amount)] = df[col] - df.shift(amount)[col]
    return df

def insertWeekday(df):
    df['dayOfWeek'] = df['timestamp'].dt.dayofweek
    return df