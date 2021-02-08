from influxdb import InfluxDBClient
import requests

import datetime
import time

import math
import pprint
import os
import signal
import sys

client = None
dbname = 'formankovi'
measurement = 'senzory_bojler'

url_request = "192.168.1.144"


def db_exists():
    '''returns True if the database exists'''
    dbs = client.get_list_database()
    for db in dbs:
        if db['name'] == dbname:
            return True
    return False

def wait_for_server(host, port, nretries=5):
    '''wait for the server to come online for waiting_time, nretries times.'''
    url = 'http://{}:{}'.format(host, port)
    waiting_time = 1
    for i in range(nretries):
        try:
            requests.get(url)
            return 
        except requests.exceptions.ConnectionError:
            print('waiting for', url)
            time.sleep(waiting_time)
            waiting_time *= 2
            pass
    print('cannot connect to', url)
    sys.exit(1)

def connect_db(host, port, reset):
    '''connect to the database, and create it if it does not exist'''
    global client
    print('connecting to database: {}:{}'.format(host,port))
    client = InfluxDBClient(host, port, retries=5, timeout=1)
    wait_for_server(host, port)
    create = False
    if not db_exists():
        create = True
        print('creating database...')
        client.create_database(dbname)
    else:
        print('database already exists')
    client.switch_database(dbname)
    if not create and reset:
        client.delete_series(measurement=measurement)

 
def measure(nmeas):
    '''insert dummy measurements to the db.
    nmeas = 0 means : insert measurements forever. 
    '''
    bad_request_sleeping_time = 20

    while True:
        try:
            http = requests.get("http://" + url_request + "/status")
            bad_request_sleeping_time = 10
        except:
            print("unable to get request")
            if(bad_request_sleeping_time != 200):
                bad_request_sleeping_time +=10
            time.sleep(bad_request_sleeping_time)
            continue  
        if http.status_code == 200:
            data = http.json()

            power = data['meters'][0]["power"]
            temperature_1 = data['ext_temperature']['0']['tC']
            temperature_2 = data['ext_temperature']['1']['tC']
            current_time = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

            data_to_db_json = [
                {
                    "measurement": measurement,
                    "time": current_time,
                    
                    "fields": {
                        "power": power,
                        "tmp1": temperature_1, 
                        "tmp2": temperature_2,
                        "turned": data['relays'][0]['ison']
                    }
                }
            ]

                    
            client.write_points(data_to_db_json)
           
        time.sleep(20)

def get_entries():
    '''returns all entries in the database.'''
    results = client.query('select * from {}'.format(measurement))
    # we decide not to use the x tag
    return list(results[(measurement, None)])

    
if __name__ == '__main__':
    import sys
    
    from optparse import OptionParser
    parser = OptionParser('%prog [OPTIONS] <host> <port>')
    parser.add_option(
        '-r', '--reset', dest='reset',
        help='reset database',
        default=False,
        action='store_true'
        )
    parser.add_option(
        '-n', '--nmeasurements', dest='nmeasurements',
        type='int', 
        help='reset database',
        default=0
        )
    
    options, args = parser.parse_args()
    if len(args)!=2:
        parser.print_usage()
        print('please specify two arguments')
        sys.exit(1)
    host, port = args
    connect_db(host, port, options.reset)
    def signal_handler(sig, frame):
        print()
        print('stopping')
        pprint.pprint(get_entries())
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)

    measure(options.nmeasurements)
        
    pprint.pprint(get_entries())