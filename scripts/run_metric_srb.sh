ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")

EXP_DIR=$1
CONFIG_DIR=$EXP_DIR/*.toml
GT_MESHES_DIR=/pv/SPIN/data/deep_geometric_prior_data
DATA_DIR=/pv/SPIN/data/deep_geometric_prior_data
RAW_DATA_DIR=/pv/SPIN/data/deep_geometric_prior_data

python3 train/compute_metrics_srb.py --config $CONFIG_DIR --log_dir $EXP_DIR --model_dir $MODEL_DIR --data_dir $DATA_DIR
