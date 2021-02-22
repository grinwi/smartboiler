import json
import requests
from influxdb import InfluxDBClient
from influxdb import DataFrameClient
import pandas as pd

import datetime
import time
import math
import pprint
import os
import signal
import sys  
import pickle
import os.path
from event_checker import EventChecker
from settings_loader import SettingsLoader




sleep_time = 120

def turn(order):
    try:        
        requests.get("http://" + socket_url +"/relay/0?turn=" + order)        
        sleep_time = 120

        return True
    except:
        print("unable to turn on/off socket")
        if(sleep_time < 500):
            sleep_time += 10
        return False

def handle(temperature_1, temperature_2, socket_turned_on):

    current_time = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

    if (EventChecker.check_event()):
        if (socket_turned_on):
            if (turn('off')):
                    print(current_time)
                    print("heating turning off because of scheduled holiday")

        else:
            while(EventChecker.check_event()):
                
                time.sleep(300)

    else:
        if ( is_time_between(datetime.time(6,30), datetime.time(7,00)) or is_time_between(datetime.time(17,30), datetime.time(21,30)) ):
            if ((temperature_2 < 36) and (socket_turned_on == False)):
                if (turn('on')):
                    print(current_time)
                    print("heating turning on in high")
            elif( (temperature_2 > 38) and (socket_turned_on == True)):
                if (turn('off')):
                    print(current_time)
                    print("heating turning off in high")
        else:
            if ((temperature_2 < 31) and (socket_turned_on == False)):
                if(turn('on')):
                    print(current_time)
                    print("heating turning on in low")
            elif( (temperature_2 > 33) and (socket_turned_on == True)):
                if(turn('off')):
                    print(current_time)
                    print("heating turning off in low")

def is_time_between(begin_time, end_time):
    # If check time is not given, default to current UTC time
    check_time = datetime.datetime.now().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time


if __name__ == "__main__":

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

    if len(args) < 2:
        parser.print_usage()
        print('please specify three arguments')
        sys.exit(1)

    host, port = args

    client = InfluxDBClient(host, port, retries=5, timeout=1)

    EventChecker = EventChecker()


    SettingsLoader = SettingsLoader(settings_file)
    settings = SettingsLoader.load_settings()

    socket_url = settings['socket_url']
    db_name = settings['db_name']
    measurement = settings['measurement']

    while(1):





        try:
            results = client.query('SELECT * FROM "' + db_name + '"."autogen"."' + measurement + '" ORDER BY DESC LIMIT 1')

        except:
            print("unable to read from db")
            continue
        tmp1 = results.raw['series'][0]['values'][0][2]
        tmp2 = results.raw['series'][0]['values'][0][3]
        socket_turned_on = results.raw['series'][0]['values'][0][4]
      
        handle(tmp1, tmp2, socket_turned_on)

        time.sleep(sleep_time)
    