ROOT_DIR= $(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")

CONFIG_DIR=$ROOT_DIR'configs/siren_2d.toml' # change to your config file path
IDENTIFIER='curv'                                                   # change to your desired identifier
LOG_DIR='./log/2d_curv/'                                           # change to your desired log path
mkdir -p $LOG_DIR

for SHAPE_TYPE in 'house' 'target' 'snowflake' 'peace' 'circle' 'seaurchin' 'snake' 'button' 'bearing' 'L' 'starhex' 'boomerangs' 'fragments' 'square' ; do # for SHAPE_TYPE in 'circle' 'L' 'square' 'snowflake' 'starhex' 'button' 'target' 'bearing' 'snake' 'seaurchin' 'peace' 'boomerangs' 'fragments' 'house'; do
    cp -r scripts/$THIS_FILE $LOG_DIR
    cp -r $CONFIG_DIR $LOG_DIR
    echo $SHAPE_TYPE
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE
done
