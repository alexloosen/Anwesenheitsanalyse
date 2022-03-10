import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def encodeCyclical(df, col, max_val):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_val)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_val)
    return df

def detectOutliers(df):
    x = df['co2_ppm']
    q1 = np.percentile(x, 5)
    q3 = np.percentile(x, 95)
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
    df = df.assign(second=lambda d: (d['timestamp'].dt.hour.astype('int') * 3600 + 
                                   d['timestamp'].dt.minute.astype('int') * 60 + 
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

def preProcessDataset(df):
    # Daten mit falschen PIR-Werten rausschmeissen
    df_new = dropSection(df, 1634518800, 1637586000)

    # Timestamp einfacher zu verarbeiten, wenn als Integer gespeichert
    df_new = convertTimestamp(df_new)
    # Integer Timestamp jetzt zyklisch encodieren
    df_new = encodeCyclical(df_new, 'second', 86400)

    # Deltas einfuegen
    df_new = addDelta(df_new, 'co2_ppm', 1)
    df_new = addDelta(df_new, 'co2_ppm', 2)
    df_new = addDelta(df_new, 'co2_ppm', 3)
    df_new = addDelta(df_new, 'co2_ppm', 4)
    df_new = addDelta(df_new, 'co2_ppm', 5)
    df_new = addDelta(df_new, 'co2_ppm', 6)
    df_new = addDelta(df_new, 'co2_ppm', 7)
    df_new = addDelta(df_new, 'co2_ppm', 8)
    df_new = addDelta(df_new, 'co2_ppm', 9)
    df_new = addDelta(df_new, 'co2_ppm', 10)
    df_new = addDelta(df_new, 'co2_ppm', 11)
    df_new = addDelta(df_new, 'co2_ppm', 12)
    df_new = addDelta(df_new, 'co2_ppm', 13)
    df_new = addDelta(df_new, 'co2_ppm', 14)
    df_new = addDelta(df_new, 'co2_ppm', 15)
    
#    df_new = addDelta(df_new, 'temperature_celsius', 1)
#    df_new = addDelta(df_new, 'temperature_celsius', 2)
#    df_new = addDelta(df_new, 'temperature_celsius', 3)
#    df_new = addDelta(df_new, 'temperature_celsius', 4)
#    df_new = addDelta(df_new, 'temperature_celsius', 5)
#    df_new = addDelta(df_new, 'temperature_celsius', 6)
#    df_new = addDelta(df_new, 'temperature_celsius', 7)
#    df_new = addDelta(df_new, 'temperature_celsius', 8)
    
#    df_new = addDelta(df_new, 'relative_humidity_percent', 1)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 2)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 3)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 4)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 5)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 6)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 7)
#    df_new = addDelta(df_new, 'relative_humidity_percent', 8)
    
    # Daten shiften
    #df_new['co2_ppm_last'] = df_new.shift(1)['co2_ppm']

    # Werte von vor n-Minuten einfuegen
#    df_new['co2_ppm_shift1'] = df_new.shift(1)['co2_ppm']
#    df_new['co2_ppm_shift2'] = df_new.shift(2)['co2_ppm']
#    df_new['co2_ppm_shift3'] = df_new.shift(3)['co2_ppm']
#    df_new['co2_ppm_shift4'] = df_new.shift(4)['co2_ppm']
#    df_new['co2_ppm_shift5'] = df_new.shift(5)['co2_ppm']
#    df_new['co2_ppm_shift6'] = df_new.shift(6)['co2_ppm']
#    df_new['co2_ppm_shift7'] = df_new.shift(7)['co2_ppm']
#    df_new['co2_ppm_shift8'] = df_new.shift(8)['co2_ppm']
#    df_new['co2_ppm_shift9'] = df_new.shift(9)['co2_ppm']
#    df_new['co2_ppm_shift10'] = df_new.shift(10)['co2_ppm']
#    df_new['co2_ppm_shift11'] = df_new.shift(11)['co2_ppm']
#    df_new['co2_ppm_shift12'] = df_new.shift(12)['co2_ppm']
#    df_new['co2_ppm_shift13'] = df_new.shift(13)['co2_ppm']
#    df_new['co2_ppm_shift14'] = df_new.shift(14)['co2_ppm']
#    df_new['co2_ppm_shift15'] = df_new.shift(15)['co2_ppm']

#    df_new['shift1_delta'] = df_new.shift(1)['co2_ppm'] - df_new.shift(2)['co2_ppm']
#    df_new['shift2_delta'] = df_new.shift(2)['co2_ppm'] - df_new.shift(3)['co2_ppm']
#    df_new['shift3_delta'] = df_new.shift(3)['co2_ppm'] - df_new.shift(4)['co2_ppm']
#    df_new['shift4_delta'] = df_new.shift(4)['co2_ppm'] - df_new.shift(5)['co2_ppm']
#    df_new['shift5_delta'] = df_new.shift(5)['co2_ppm'] - df_new.shift(6)['co2_ppm']
#    df_new['shift6_delta'] = df_new.shift(6)['co2_ppm'] - df_new.shift(7)['co2_ppm']

    # Wochentag einfuegen
    # verringert Genauigkeit, weil wahrscheinlich zu "verlaesslich"

    df_new = insertWeekday(df_new)
    #df_new = df_new.drop(df_new[df_new.dayOfWeek > 4].index)

    # Leere Felder entfernen
    df_new = df_new.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # Ausreisser mit Interquartile Range (IQR) und Tukey's Method loeschen
    outlier_indices = detectOutliers(df_new)
    df_new.drop(index=outlier_indices, inplace=True)

    return df_new

def reshape_data_for_LSTM(X, y, timesteps_per_sample):
    X = X.copy()
    sample_count = int(X.count()[0]/timesteps_per_sample)
    
    while X.count()[0] % sample_count != 0:
        X = X.iloc[1: , :]
        y = y.iloc[1:]
    
    X_array = X.to_numpy()
    y_array = y.to_numpy()
    
    feature_count = len(X.columns)    
    
    print('sample_count = ' + str(sample_count))
    print('timesteps_per_sample = ' + str(timesteps_per_sample))
    print('feature_count = ' + str(feature_count))
    
    X_reshaped = np.reshape(X_array,[sample_count, timesteps_per_sample, feature_count]) 
    y_reshaped = np.reshape(y_array,[sample_count, timesteps_per_sample, 1])
    
    return X_reshaped, y_reshaped

def normalize_min_max(dataframe, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))       
        dataframe = pd.DataFrame(data=scaler.fit_transform(dataframe), columns=dataframe.columns)
    else:
        dataframe = pd.DataFrame(data=scaler.transform(dataframe), columns=dataframe.columns)
    return dataframe, scaler