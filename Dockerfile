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
# copy contents
COPY src/smartboiler/ /app/src/smartboiler/


COPY src/smartboiler/optimization.py /app/src/smartboiler/optimization.py
COPY src/smartboiler/retrieve_hass.py /app/src/smartboiler/retrieve_hass.py
COPY src/smartboiler/utils.py /app/src/smartboiler/utils.py
COPY src/smartboiler/web_server.py /app/src/smartboiler/web_server.py
COPY src/smartboiler/templates/index.html /app/src/smartboiler/templates/index.html
COPY src/smartboiler/static/style.css /app/src/smartboiler/static/style.css

COPY options.json /app/

COPY config_smartboiler.yaml /app/config_smartboiler.yaml
COPY secrets_smartboiler.yaml /app/secrets_smartboiler.yaml



# CMD ["python", "-u", "control.py","-f", "settings.json"]
RUN python3 setup.py install

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# configure the container to run in an executed manner
CMD [ "python3", "src/smartboiler/controller.py" ]
