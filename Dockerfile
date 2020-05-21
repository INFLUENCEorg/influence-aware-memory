FROM python:3.6-stretch
MAINTAINER Miguel Suau <miguel.suau@gmail.com>

ENV SUMO_HOME="$PWD/sumo"
ENV PYTHONPATH="${PYTHONPATH}:${SUMO_HOME}/tools"

# Sumo install
RUN apt-get update -y \
    && apt-get install -y cmake python g++ libxerces-c-dev libfox-1.6-dev libgdal-dev libproj-dev libgl2ps-dev swig \
    && git clone --recursive https://github.com/eclipse/sumo \
    && mkdir sumo/build/cmake-build && cd sumo/build/cmake-build \
    && cmake ../.. \
    && make -j$(nproc)


# Install python dependencies
RUN pip3 install --upgrade pip
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/openai/baselines.git \
    && cd baselines \
    && pip3 install -e .

# check our python environment
RUN python3 --version
RUN pip3 --version

# set the working directory for containers
WORKDIR ./

# Installing python dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy all the files from the project’s root to the working directory
COPY ./ ./
