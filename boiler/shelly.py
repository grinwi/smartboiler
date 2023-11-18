import requests
import time
import json
import pandas as pd
from datetime import datetime
    
class Shelly:
    def __init__ (self, server_url, auth_token, device_id):
        self.server_url = server_url
        self.auth_token = auth_token
        self.device_id = device_id
        self.ison = False
        
    def turn_on(self):
        #turn on device
        pass
    def turn_off(self):
        #turn off device
        pass
    def get_device_status(self):
        bad_request_sleeping_time = 5
        try:
            
            #create http post request with use of request library to get data with argument of device id and auth key
            print("creating request")
            
            http_post_request = requests.post(self.server_url + "/device/status", data = {"id": self.device_id, "auth_key": self.auth_token})
            print("request created")
        except requests.exceptions.RequestException as e:
            #if request is not successful, print error and sleep for 5 seconds
            print(e)

            if(bad_request_sleeping_time != 200):
                bad_request_sleeping_time +=5
            time.sleep(bad_request_sleeping_time)
            return
              
        if http_post_request.status_code == 200:
            #if request is successful, return data
            usable_data = {}
            data =  http_post_request.json()
            data = data['data']['device_status']
            usable_data['power'] = data['meters'][0]["power"]
            self.ison = bool(data['relays'][0]['ison'])
            for i in range(0, len(data['ext_temperature'])):
                usable_data['tmp' + str(i)] = data['ext_temperature'][i]['tC']
            usable_data['time'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
            usable_data['ison'] = self.ison
            usable_data['device_tmp'] = data['tmp']['tC']
            self.shelly_id = data['getinfo']["fw_info"]["device"]
            usable_data['shelly_id'] = self.shelly_id
            print(usable_data)
            return usable_data
    def get_data():
        # get data from shelly cloud
        pass
        # return data

class SmartBoiler(Shelly):
    def __init__(self, server_url, auth_token, device_id, volume=80, max_temp=70, wattage=2000, min_tmp=37, one_shower_volume=40, shower_tmp=40):
        super().__init__(server_url, auth_token, device_id)
        self.volume = volume
        self.max_temp = max_temp
        self.wattage = wattage
        self.min_tmp = min_tmp
        self.one_shower_volume = one_shower_volume
        self.shower_tmp = shower_tmp

if __name__ == "__main__":

    server_url = "https://shelly-63-eu.shelly.cloud"
    auth_token = "MTgwMGY3dWlk7121FEEFF87F657946F5BA357BFF57313F2320187ED2CB88C2C46BD89E5E52459B9010E678CFBE44"
    shelly_device_id = "a4cf12f3f0cd"
    ShellyClass = SmartBoiler(server_url, auth_token, shelly_device_id)
    ShellyClass.get_device_status()