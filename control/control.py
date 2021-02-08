import requests
from influxdb import InfluxDBClient

import datetime
import time

import math
import pprint
import os
import signal
import sys

got_data = False
sleep_time = 120

print("waiting..")
#time.sleep(30)
request_url = "192.168.1.144"

def turn(on):
    try:
        if on:
            requests.get("http://" + request_url +"/relay/0?turn=on")
        else:
            requests.get("http://" + request_url + "/relay/0?turn=off")  
        sleep_time = 120
        return True
    except:
        print("unable to turn on/off socket")
        if(sleep_time < 500):
            sleep_time += 10
        return False

def handle(temperature_1, temperature_2, socket_turned_on):
    if ( is_time_between(datetime.time(5,30), datetime.time(6,00)) or is_time_between(datetime.time(16,30), datetime.time(20,30)) ):
        if ((temperature_2 < 36) and (socket_turned_on == False)):
            if (turn(True)):
                print(current_time)
                print("heating turning on in high")
        elif( (temperature_2 > 38) and (socket_turned_on == True)):
            if (turn(False)):
                print(current_time)
                print("heating turning off in high")
    else:
        if ((temperature_2 < 31) and (socket_turned_on == False)):
            if(turn(True)):
                print(current_time)
                print("heating turning on in low")
        elif( (temperature_2 > 33) and (socket_turned_on == True)):
            if(turn(False)):
                print(current_time)
                print("heating turning off in low")

def is_time_between(begin_time, end_time):
    # If check time is not given, default to current UTC time
    check_time = datetime.datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

client = InfluxDBClient("influxdb", 8086, retries=5, timeout=1)


while(1):
    try:
        results = client.query('SELECT * FROM "formankovi"."autogen"."senzory_bojler" ORDER BY DESC LIMIT 1')

    except:
        print("unable to read from db")
    tmp1 = results.raw['series'][0]['values'][0][2]
    tmp2 = results.raw['series'][0]['values'][0][3]
    socket_turned_on = results.raw['series'][0]['values'][0][4]

    try:
        handle(tmp1, tmp2, socket_turned_on)
    except:
        print("unable to handle a switch")
    time.sleep(sleep_time)
    