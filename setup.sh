#!/bin/bash

virtualenv .env -p python3.7
source .env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
# pip install .