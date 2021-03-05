# Use an official Python runtime as a parent image
FROM python

# Set the working directory to /app
WORKDIR /app
COPY turning_on/requirements.txt /app/requirements.txt
# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY turning_on/turning_on.py /app/turning_on.py



# Run app.py when the container launches
# The -u flag specifies to use the unbuffered ouput.
# in this way, what's printed by the app is visible on the host
# while the container is running
CMD ["python", "-u", "turning_on.py"]
