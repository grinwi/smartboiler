###########################################################
# Bachelor's thesis                                       #
# From a dumb boiler to a smart one using a smart socket  #
# Author: Adam Grünwald                                   #
# BUT FIT BRNO, Faculty of Information Technology         #
# 26/6/2021                                               #
#                                                         #
# Module that collects data from a smart socket.          #
###########################################################

from influxdb import InfluxDBClient
import requests

import datetime
import time
import json

import math
import pprint
import os
import signal
import sys
from settings_loader import SettingsLoader
from event_checker import EventChecker

EvntChecker = EventChecker()


client = None

def db_exists():
    """Returns True if the database exists.

    Returns:
        [boolean]: [True if DB exists]
    """
    dbs = client.get_list_database()
    for db in dbs:
        if db['name'] == db_name:
            return True
    return False

def wait_for_server(nretries=5):
    """Waiting to server response

    Args:
        host ([string]): [host]
        port ([int]): [port number]
        nretries (int, optional): [number of retries]. Defaults to 5.
    """
    url = 'http://{}:{}'.format(host, port)
    waiting_time = 1
    for i in range(nretries):
        try:
            requests.get(url)
            return 
        except requests.exceptions.ConnectionError:
            print('waiting for', url)
            time.sleep(waiting_time)
            waiting_time += 2
            pass
    print('cannot connect to', url)

def connect_to_db():
    global client
    print('connecting to database: {}:{}'.format(host,port))
    client = InfluxDBClient(host, port, retries=5, timeout=1)
    wait_for_server()
    if not db_exists():
        print('creating database...')
        client.create_database(db_name)
    else:
        print('database already exists')
    client.switch_database(db_name)


 
def measure(bad_request_sleeping_time):

    try:
        http = requests.get("http://" + socket_url + "/status")
        bad_request_sleeping_time = 10
    except:
        print("unable to get request")
        if(bad_request_sleeping_time != 200):
            bad_request_sleeping_time +=10
        time.sleep(bad_request_sleeping_time)
        return  
    if http.status_code == 200:
        try:
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
                        "turned": data['relays'][0]['ison'],
                        "in_event" : EvntChecker.check_off_event()
                    }
                }
            ]

                    
        except:
            print("unable to read data")
        try:
            client.write_points(data_to_db_json)
        except:
            print("unable to write data in db")
        
    time.sleep(20)

def load_settings(file_name):
    try:
        with open(file_name) as json_file:
            data = json.load(json_file)
    except:
        print("Error loading settings")
        return None
    
    return data["settings"]


    
if __name__ == '__main__':
    import sys
    
    from optparse import OptionParser

    parser = OptionParser('%prog [OPTIONS] <host> <port>')

  
    parser.add_option(
        '-f', '--settings_file', dest='settings_file',
        type='string', 
        default=None
        )
    options, args = parser.parse_args()

    settings_file = options.settings_file
    if len(args)< 2:
        parser.print_usage()
        print('please specify two or more arguments')
        sys.exit(1)        
    host, port = args


    SettingsLoader = SettingsLoader(settings_file)
    settings = SettingsLoader.load_settings()        

    socket_url = settings['socket_url']
    db_name = settings['db_name']
    measurement = settings['measurement']

    connect_to_db()
    bad_request_sleeping_time = 20


    while(True):
        measure(bad_request_sleeping_time)
        
