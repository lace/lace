FROM python:2.7.15

WORKDIR /src

RUN apt-get update
RUN apt-get install -y --no-install-recommends libsuitesparse-dev libboost-dev
RUN rm -rf /var/lib/apt/lists/*

# numpy is an install dependency for blmath and lace.
RUN pip install numpy==1.13.1

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY . /src

# Set up Python environment.
ENV PYTHONPATH /src

ENTRYPOINT ["/usr/local/bin/python"]
