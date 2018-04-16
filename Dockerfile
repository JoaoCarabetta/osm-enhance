FROM python:3-alpine

# install dependencies
# the lapack package is only in the community repository
RUN echo "http://dl-4.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories
RUN apk --update add --no-cache \
    lapack-dev \
    gcc \
    freetype-dev

# Install dependencies
RUN apk add --no-cache --virtual .build-deps \
    gfortran \
    musl-dev \
    g++

RUN ln -s /usr/include/locale.h /usr/include/xlocale.h

RUN echo "http://mirror.leaseweb.com/alpine/edge/testing" >> /etc/apk/repositories
RUN apk add --no-cache geos-dev
RUN apk add --no-cache gdal-dev


WORKDIR .

RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scipy
RUN pip3 install matplotlib
RUN pip3 install osmnx
RUN pip3 install networkx
RUN pip3 install rtree
RUN pip3 install tpot
RUN pip3 install sklearn

RUN apk add --no-cache curl
RUN apk add --no-cache make
RUN apk add --no-cache git
RUN apk add --no-cache cmake

RUN echo "**** compile spatialindex ****" && \
 git clone https://github.com/libspatialindex/libspatialindex /tmp/spatialindex && \
 cd /tmp/spatialindex && \
 cmake . \
	-DCMAKE_INSTALL_PREFIX=/usr && \
 make && \
 make && \
 make install

COPY . .

CMD [ "python3", "./osm_enhance.py" ]