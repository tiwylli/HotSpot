#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")"))) 
cd $DIR/SPIN/data/

# Install gdown if not already installed
pip install --user gdown

# Get nie from Google Drive
gdown --id 12Z0nmNISGlsfy-QK3_uw91uuj9hC6DDc -O NIE.zip
unzip NIE.zip
