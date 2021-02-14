# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app

# Install any needed packages specified in requirements.txt
COPY sensor/requirements.txt /app/requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY sensor/sensor.py /app/sensor.py
COPY settings_loader/settings_loader.py /app/settings_loader.py

COPY settings.json /app/settings.json
# Run app.py when the container launches
# The -u flag specifies to use the unbuffered ouput.
# in this way, what's printed by the app is visible on the host
# while the container is running
CMD ["python", "-u", "sensor.py", "influxdb", "8086"]
