ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")

CONFIG_DIR=$ROOT_DIR'configs/spin_2d.toml' # Change to your config file path
LOG_DIR=$ROOT_DIR'log/2D/'                 # Change to your log path
IDENTIFIER='StEik2D-2024-10-10-12-12-15'   # change to your desired identifier
CONFIG_DIR=$LOG_DIR/$IDENTIFIER/*.toml
EXP_DIR=$LOG_DIR$IDENTIFIER/

python3 train/compute_metrics_2d.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER --model_dir $MODEL_DIR
