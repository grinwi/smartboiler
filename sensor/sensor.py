###########################################################
# Bachelor's thesis                                       #
# From a dumb boiler to a smart one using a smart socket  #
# Author: Adam Grünwald                                   #
# BUT FIT BRNO, Faculty of Information Technology         #
# 26/6/2021                                               #
#                                                         #
# Module that collects data from a smart socket.          #
###########################################################

import influxdb_client, os, time
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import requests
client = None
import os
import sys
import math
import json
import time
import pprint
import signal
import datetime
from shelly import SmartBoiler
from optparse import OptionParser
from settings_loader import SettingsLoader

db_token = "u8SqwalMUxwyO_AMraGkTWdmp04gbdcboUN3tlEBzMd9xvkPg0cB3MAOsMCwZV-Iy7i2DT_VaLDYjQ-NM_a_LA=="
bucket = "smart_boiler" 
org = "smart_household"
url = "http://influxdb:8086"
write_client = InfluxDBClient(url=url, token=db_token, org=org)
# from event_checker import EventChecker

# EvntChecker = EventChecker()

# def db_exists():
#     """Returns True if the database exists.

#     Returns:
#         [boolean]: [True if DB exists]
#     """
#     dbs = client.get_list_database()
#     for db in dbs:
#         if db['name'] == db_name:
#             return True
#     return False

# def wait_for_server(nretries=5):
#     """Waiting to server response

#     Args:
#         host ([string]): [host]
#         port ([int]): [port number]
#         nretries (int, optional): [number of retries]. Defaults to 5.
#     """
#     url = 'http://{}:{}'.format(host, port)
#     waiting_time = 1
#     for i in range(nretries):
#         try:
#             requests.get(url)
#             return 
#         except requests.exceptions.ConnectionError:
#             print('waiting for', url)
#             time.sleep(waiting_time)
#             waiting_time += 2
#             pass
#     print('cannot connect to', url)

# def connect_to_db():
#     global client
#     print('connecting to database: {}:{}'.format(host,port))
#     client = InfluxDBClient(host, port, retries=5, timeout=1)
#     wait_for_server()
#     if not db_exists():
#         print('creating database...')
#         client.create_database(db_name)
#     else:
#         print('database already exists')
#     client.switch_database(db_name)


 
def measure(bad_request_sleeping_time):
    measurement = SmartBoiler.get_device_status()
    try:
        #client.write_points(data_to_db_json)
        #####
        write_api = write_client.write_api(write_options=SYNCHRONOUS)
        print(write_client)
        print('creating point')
        point = Point(measurement)\
            .tag("shelly_id", measurement['shelly_id'])\
            .field("power", measurement['power'])\
            .field("tmp0", measurement['tmp0'])\
            .field("tmp1", measurement['tmp1'])\
            .field("ison", measurement['ison'])\
            .field("in_event", False)\
            .time(datetime.datetime.utcnow())

        print('point created')
        print(point)
        print('writing point to db')

        write_api.write(bucket=bucket, org=org, record=point)
        time.sleep(1) # separate points by 1 second
    #catch except and print out the error
    except Exception as e:
        print(e)
    
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



   
    #connect_to_db()
    bad_request_sleeping_time = 20

    server_url = "https://shelly-63-eu.shelly.cloud"
    auth_token = "MTgwMGY3dWlk7121FEEFF87F657946F5BA357BFF57313F2320187ED2CB88C2C46BD89E5E52459B9010E678CFBE44"
    shelly_device_id = "a4cf12f3f0cd"
    SmartBoiler = SmartBoiler(server_url, auth_token, shelly_device_id)


    while(True):
        measure(bad_request_sleeping_time)
        
