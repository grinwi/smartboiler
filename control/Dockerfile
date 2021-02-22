# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY control/requirements.txt /app/requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

COPY control/control.py /app/control.py
COPY control/bojler.py /app/bojler.py
COPY control/time_handler.py /app/time_handler.py
COPY control/week_planner.py /app/week_planner.py
COPY control/event_checker.py /app/event_checker.py

COPY settings.json /app/settings.json
#VOLUME /app

COPY settings_loader/settings_loader.py /app/settings_loader.py

COPY control/credentials.json /app/credentials.json
COPY control/token.pickle /app/token.pickle

# Run app.py when the container launches
# The -u flag specifies to use the unbuffered ouput.
# in this way, what's printed by the app is visible on the host
# while the container is running
CMD ["python", "-u", "control.py","-f", "settings.json"]
