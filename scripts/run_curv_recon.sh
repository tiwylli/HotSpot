ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")

CONFIG_DIR=$ROOT_DIR'configs/curv_recon.toml' # change to your config file path
IDENTIFIER='spin_curv'                                                   # change to your desired identifier
LOG_DIR='./log/2d_curv/'                                           # change to your desired log path
mkdir -p $LOG_DIR

for SHAPE_TYPE in 'boomerangs' 'fragments' 'house' 'seaurchin' 'target' 'circle' 'L' 'square' 'snowflake' 'starhex' 'button'  'bearing' 'snake' 'peace' ; do # for SHAPE_TYPE in 'circle' 'L' 'square' 'snowflake' 'starhex' 'button' 'target' 'bearing' 'snake' 'seaurchin' 'peace' 'boomerangs' 'fragments' 'house'; do
    cp -r scripts/$THIS_FILE $LOG_DIR
    cp -r $CONFIG_DIR $LOG_DIR
    echo $SHAPE_TYPE
    python3 train/train.py --config $CONFIG_DIR --log_dir $EXP_DIR/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE --saved_model_dir $SAVED_MODEL_DIR
done
