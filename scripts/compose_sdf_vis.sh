#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./scripts/compose_sdf_vis.sh <folder_name>"
    exit 1
fi

folder_name=$1
python3 utils/compose_sdf_vis.py $folder_name
