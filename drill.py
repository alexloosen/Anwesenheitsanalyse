import requests
import pandas as pd
import datetime as dt

def connect_drill(query, caching=True, chunk_size: int = 0):
    #username = os.getenv("DRILLUSERNAME")
    host = 'https://proxima.bigdata.fh-aachen.de:8047'
    username = 'al7739s'
    password = 'tWtx4UYhTdUbPHumX3VixMhdi'
    headers = {'Content-Type': 'application/json',
               'Authorization': '%s:%s' % (username, password)}
    #headers = {'Authorization': username + ':' + password}
    if caching:
        headers["Cache-Control"] = "max-age=" + "1440"
    else:
        headers["Cache-Control"] = "max-age=" + "0"
    #if chunk_size > 0:
        #headers["format"] = "chunks:" + str(chunk_size)
    data = {'query': "{q}".format(q=query)}

    try:
        result = requests.post(host + '/query', json=data, headers=headers, verify=True)
        print(result)
    except Exception as e:
        print("The drill-proxy is not reachable. Please check if you are in the FH-Aachen network.")
        raise (e)

    data = None
    try:
        data = pd.read_json(result.text)
        if data.empty:
            print('Result of query is empty!')
            print('Query was: ' + query)
    except ValueError:
        print("Something went wrong when converting the json string from the datasource to a pandas DataFrame.")
        print(result.text)
    return data
    
#    query = """SELECT *
#                FROM dfs.co2meter.`sensor_data`
#                WHERE `serial_number` = 's_3c6105d3abae_299589'
#                FETCH FIRST 100000 ROWS ONLY"""

#    query = """SELECT `timestamp`,`room`, `presence`, `co2_ppm`, `temperature_celsius`, `relative_humidity_percent` 
#                FROM ipenv.data.`sensor_data_v1`
#                WHERE `timestamp` > 1627776000
#                AND `room` LIKE '{room}'
#                LIMIT 1000000""".format(room=room)

#                WHERE SensorID LIKE 's_e8db84c5f33d_281913'

#{"room":{"0":"Daniel","1":"Felix N #2","2":"Calvin","3":"bigDataLab","4":"FelixAkku","5":"Galla","6":null,"7":"Lukasbuero","8":"Remmy","9":"Felix B. #1","10":"Felix N #1","11":"Elsen","12":"Internal Server Error"}}


def get_PIR_data(room: str = "H217", presence = True):
    dict_rooms = {'dfs': 'dfs', 'H217': 'Elsen', 'H216': 'Galla', 'H215': 'Remmy', '0':'Daniel',
                  '1':'Felix N#2','2':'Calvin','3':'bigDataLab','4':'FelixAkku', '7':'Lukasbuero','9':'Felix B. #1','10':'Felix N #1'}
    room = dict_rooms[room]
    
    query = """SELECT *
                FROM ipenv.data.`sensor_data_v1`
                WHERE `timestamp` > 1627776000
                AND `room` LIKE '{room}' LIMIT 100""".format(room=room)
#                LIMIT 1000000""".format(room=room)

    pir_data = pd.DataFrame
    pir_data = connect_drill(query, caching=True)
    
    if (pir_data.empty):
        return pir_data

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    # apply CET time offset to timestamp
    pir_data["timestamp"].dt.tz_localize('Europe/Berlin', ambiguous=True, nonexistent='shift_forward')
    pir_data["timestamp"] = pir_data["timestamp"] + pd.Timedelta(hours=2)

    if (presence):
        pir_data["presence"] = pir_data["presence"].astype(int)

    #pir_data = pir_data.groupby(pd.Grouper(key="timestamp", freq="2min")).mean()\
    #    .round(0).reset_index(drop=False)

    return pir_data