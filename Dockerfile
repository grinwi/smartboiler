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
COPY src/smartboiler/control.py /app/control.py
COPY src/smartboiler/boiler.py /app/boiler.py
COPY src/smartboiler/switch.py /app/switch.py
COPY src/smartboiler/time_handler.py /app/time_handler.py
COPY src/smartboiler/week_planner.py /app/week_planner.py
COPY src/smartboiler/event_checker.py /app/event_checker.py
COPY src/smartboiler/web_server.py /app/src/smartboiler/web_server.py

COPY src/smartboiler/settings.json /app/settings.json
COPY src/smartboiler/templates/index.html /app/src/smartboiler/templates/index.html
COPY src/smartboiler/static/style.css /app/src/smartboiler/static/style.css

# CMD ["python", "-u", "control.py","-f", "settings.json"]
RUN python3 setup.py install

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# configure the container to run in an executed manner
CMD [ "python3", "src/smartboiler/control.py" ]

