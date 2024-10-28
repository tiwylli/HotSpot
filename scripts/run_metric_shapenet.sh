ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")

EXP_DIR=$1
CONFIG_DIR=$EXP_DIR/*.toml
GT_MESHES_DIR=/pv/SPIN/data/ShapeNetNSP-watertight-meshes
DATA_DIR=/pv/SPIN/data/NSP_dataset
RAW_DATA_DIR=/pv/SPIN/data/ShapeNetNSP

python3 train/compute_metrics_shapenet.py --config $CONFIG_DIR --log_dir $EXP_DIR --model_dir $MODEL_DIR --data_dir $DATA_DIR --raw_dataset_path $RAW_DATA_DIR --gt_meshes_dir $GT_MESHES_DIR
