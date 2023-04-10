###########################################################
# Bachelor's thesis                                       #
# From a dumb boiler to a smart one using a smart socket  #
# Author: Adam Grünwald                                   #
# BUT FIT BRNO, Faculty of Information Technology         #
# 26/6/2021                                               #
#                                                         #
# Docker file for the control mechanism of a smart boiler #
###########################################################

FROM python

# Set the working directory to /app
WORKDIR /app

# Installing requirements from a requirements file
COPY control/requirements.txt /app/requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

#installing an api for google calendar
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

# copying of files needed to run the script (modules etc...)
COPY control/control.py /app/control.py
COPY control/boiler.py /app/boiler.py
COPY control/time_handler.py /app/time_handler.py
COPY control/week_planner.py /app/week_planner.py
COPY control/event_checker.py /app/event_checker.py
COPY settings.json /app/settings.json
COPY settings_loader/settings_loader.py /app/settings_loader.py

# copy of credentials needed for google api
# COPY control/credentials.json /app/credentials.json
# COPY control/token.pickle /app/token.pickle

# run of script itself
CMD ["python", "-u", "control.py","-f", "settings.json"]
