#!/bin/bash
DIR=$(dirname $(dirname "$(readlink -f "$0")"))  # Should point to your root path
cd $DIR/data/

# Install gdown if not already installed
pip install gdown

# Get Scene from SIREN (56.2MB) (https://drive.google.com/file/d/13X1UlMsnbh3dcV4tJysVDgzg6kYyxHhb/view?usp=sharing)
gdown --id 13X1UlMsnbh3dcV4tJysVDgzg6kYyxHhb -O scene_reconstruction.tar.xz
tar -xvf scene_reconstruction.tar.xz
