FROM python:3.8-slim-buster

# switch working directory
WORKDIR /app

# copy the requirements file into the image
COPY requirements.txt requirements.txt
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
    && apt-get purge -y --auto-remove \
        gcc \
        libhdf5-dev \
        libhdf5-serial-dev \
        netcdf-bin \
        libnetcdf-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --trusted-host pypi.python.org -r requirements.txt


# copying of files needed to run the script (modules etc...)
# copy contents
COPY src/smartboiler/ /app/src/smartboiler/


RUN python3 setup.py install

# set env variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# configure the container to run in an executed manner
CMD [ "python3", "src/smartboiler/controller.py" ]