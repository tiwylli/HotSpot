ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")

<<<<<<< HEAD
LOG_DIR=$ROOT_DIR'log/2d_curv/'                 # Change to your log path
IDENTIFIER='igr-2024-10-12-17-02-38'   # change to your desired identifier
=======
LOG_DIR=$ROOT_DIR'log/2D/'                 # Change to your log path
IDENTIFIER='StEik2D-2024-10-10-12-12-15'   # Change to your desired identifier
>>>>>>> 7aeaca7d9298a8244048bbd0850554b42740fa83
CONFIG_DIR=$LOG_DIR/$IDENTIFIER/*.toml
EXP_DIR=$LOG_DIR$IDENTIFIER/

python3 train/compute_metrics_2d.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER --model_dir $MODEL_DIR
