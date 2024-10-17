ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")

EXP_DIR=$1
CONFIG_DIR=$EXP_DIR/*.toml

python3 train/compute_metrics_2d.py --config $CONFIG_DIR --log_dir $EXP_DIR --model_dir $MODEL_DIR
