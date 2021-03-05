import requests
import time
import datetime

while(1):
    try:
        requests.get("http://192.168.1.222/relay/0?turn=on")
    except:
        print("unable to turn on at {}".format(datetime.datetime.now()))

    time.sleep(20)