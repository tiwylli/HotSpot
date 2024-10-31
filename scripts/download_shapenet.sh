#!/bin/bash
DIR=$(dirname $(dirname "$(readlink -f "$0")"))  # Should point to your DiGS path
cd $DIR/data/

# Install gdown if not already installed
pip install gdown

# Get preprocessed subset of ShapeNet data (370.96MB)
# Google Drive link: https://drive.google.com/file/d/14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0/view?usp=sharing
gdown --id 14CW_a0gS3ARJsIonyqPc5eKT3iVcCWZ0 -O NSP_data.tar.gz
tar -xzvf NSP_data.tar.gz

# Get .ply point tree files for ShapeNet subset (412.80MB)
# Google Drive link: https://drive.google.com/file/d/1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf/view?usp=sharing
gdown --id 1h6TFHnza0axOZz5AuRkfyLMx_sFcu_Yf -O ShapeNetNSP.tar.gz
tar -xzvf ShapeNetNSP.tar.gz
