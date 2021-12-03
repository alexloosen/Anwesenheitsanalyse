import requests

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
        result = requests.post(host + '/query', json=data, headers=headers)
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
    
#   SELECT * FROM dfs.co2meter.`sensor_data`
#   SELECT `timestamp`,`room`, `presence`, `co2_ppm`, `temperature_celsius`, `relative_humidity_percent` 
#    FROM ipenv.data.`sensor_data`
#    WHERE `room` LIKE '{room}' 
#    AND `timestamp` > 1627776000
#    AND `timestamp` < 1634346061
def get_PIR_data(room: str = "H217"):
    dict_rooms = {'H217': 'Elsen', 'H216': 'Galla', 'H215': 'Remmy'}
    room = dict_rooms[room]
    
    query = """SELECT `timestamp`,`room`, `presence`, `co2_ppm`, `temperature_celsius`, `relative_humidity_percent` 
                FROM ipenv.data.`sensor_data`
                WHERE `room` LIKE '{room}' 
                AND `timestamp` > 1627776000
                ORDER BY `timestamp` ASC
                LIMIT 10000000""".format(room=room)
    
    pir_data = connect_drill(query, caching=True)
    
    # apply CET time offset to timestamp
    pir_data["timestamp"].dt.tz_localize('Europe/Berlin', ambiguous=True, nonexistent='shift_forward')
    pir_data["timestamp"] = pir_data["timestamp"] + pd.Timedelta(hours=2)

    pir_data["presence"] = pir_data["presence"].astype(int)
    pir_data.head()
    
    pir_data = pir_data.groupby(pd.Grouper(key="timestamp", freq="5min")).mean()\
        .round(0).reset_index(drop=False)
    
    return pir_data