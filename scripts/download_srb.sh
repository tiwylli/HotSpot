#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
cd $DIR/data/

# Install gdown if not already installed
pip install --user gdown

# get SRB data (1.12GB) (https://drive.google.com/file/d/17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe/view)
gdown --id 17Elfc1TTRzIQJhaNu5m7SckBH_mdjYSe -O deep_geometric_prior_data.zip
unzip deep_geometric_prior_data.zip
