FROM python:3.8-slim-buster

# switch working directory
WORKDIR /app

# copy the requirements file into the image
COPY requirements.txt requirements.txt
COPY requirements_webserver.txt requirements_webserver.txt
COPY setup.py setup.py
COPY README.md README.md

# Setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        libhdf5-dev \
        libhdf5-serial-dev \
        netcdf-bin \    
        libnetcdf-dev \
        coinor-cbc \
        coinor-libcbc-dev \
    && ln -s /usr/include/hdf5/serial /usr/include/hdf5/include \
    && export HDF5_DIR=/usr/include/hdf5 \
    && pip3 install netCDF4 \
    && pip3 install --no-cache-dir -r requirements_webserver.txt \
    && apt-get purge -y --auto-remove \
        gcc \
        libhdf5-dev \
        libhdf5-serial-dev \
        netcdf-bin \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir -r requirements_webserver.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt


# copying of files needed to run the script (modules etc...)
COPY src/smart_boiler/control.py /app/control.py
COPY src/smart_boiler/boiler.py /app/boiler.py
COPY src/smart_boiler/time_handler.py /app/time_handler.py
COPY src/smart_boiler/week_planner.py /app/week_planner.py
COPY src/smart_boiler/event_checker.py /app/event_checker.py
COPY src/smart_boiler/settings.json /app/settings.json
COPY src/smart_boiler/web_server.py /app/src/smart_boiler/web_server.py
COPY src/smart_boiler/templates/index.html /app/src/smart_boiler/templates/index.html
COPY src/smart_boiler/static/style.css /app/src/smart_boiler/static/style.css

# CMD ["python", "-u", "control.py","-f", "settings.json"]
RUN python3 setup.py install

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# configure the container to run in an executed manner
CMD [ "python3", "src/smart_boiler/control.py" ]

