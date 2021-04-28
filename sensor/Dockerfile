###########################################################
# Bachelor's thesis                                       #
# From a dumb boiler to a smart one using a smart socket  #
# Author: Adam Grünwald                                   #
# BUT FIT BRNO, Faculty of Information Technology         #
# 26/6/2021                                               #
#                                                         #
# Dockerfile for sensor container measuring the data      #
###########################################################

FROM python

WORKDIR /app
# copy requirements
COPY sensor/requirements.txt /app/requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# install the google api client
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# copying files needed to run of a script
COPY sensor/sensor.py /app/sensor.py
COPY settings.json /app/settings.json
COPY settings_loader/settings_loader.py /app/settings_loader.py
COPY control/time_handler.py /app/time_handler.py
COPY control/event_checker.py /app/event_checker.py
COPY control/credentials.json /app/credentials.json
COPY control/token.pickle /app/token.pickle


CMD ["python", "-u", "sensor.py", "influxdb", "8086"]
