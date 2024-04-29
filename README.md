# fasthgp
A reference implementation of fast Hilbert-space GPs.
Can utilize the apparent Hankel/Toeplitz structure in HGPs.
Can further provide very fast approximate predictions in HGPs.
This repository only contains the reference implementation. 
For usage, please see [HGP Hankel Structure](https://github.com/AOKullberg/hgp-hankel-structure) and [Adaptive BF Selection][https://github.com/AOKullberg/adaptive-bf-selection].

## Getting started via Docker
A Dockerfile is provided which should contain everything necessary to run stuff.
E.g. build by
```
docker build -t fasthgp .
```
Run container
```
docker-compose up
```
This starts up a Jupyter lab session for you that you can open in your favourite browser.
You can also run the Python scripts in this environment.

## Getting started via pipenv
If you don't want to use Docker, pipenv is quite convenient, make sure to have it installed before proceeding.
##### Install Python 3.11
```
python3 -m pipenv --python 3.11
```
##### Install requirements
```
python3 -m pipenv install -r requirements.txt -e . --skip-lock
```
##### Run environment
```
python3 -m pipenv shell
```