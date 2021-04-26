# Use an official Python runtime as a parent image
FROM python

WORKDIR /app

COPY sensor/requirements.txt /app/requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib


COPY sensor/sensor.py /app/sensor.py

COPY settings.json /app/settings.json
COPY settings_loader/settings_loader.py /app/settings_loader.py

COPY control/time_handler.py /app/time_handler.py
COPY control/event_checker.py /app/event_checker.py
COPY control/credentials.json /app/credentials.json
COPY control/token.pickle /app/token.pickle


CMD ["python", "-u", "sensor.py", "influxdb", "8086"]
