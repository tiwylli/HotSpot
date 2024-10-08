ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
# TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")

CONFIG_DIR=$ROOT_DIR'configs/basic_shapes.toml' # change to your config file path
IDENTIFIER='curv-2024-10-08-00-09-10'                          # change to your desired identifier
LOG_DIR=$ROOT_DIR'log/2d_curv/'             # change to your desired log path
mkdir -p $LOG_DIR

for SHAPE_TYPE in 'house' 'target' 'snowflake' 'peace' 'circle' 'seaurchin' 'snake' 'button' 'bearing' 'L' 'starhex' 'boomerangs' 'fragments' 'square'; do # shapes: 'L', 'circle', 'snowflake', 'starhex'
    cp -r scripts/$THIS_FILE $LOG_DIR
    cp -r $CONFIG_DIR $LOG_DIR
    echo $SHAPE_TYPE
    SAVED_MODEL_DIR=$LOG_DIR/$IDENTIFIER/$SHAPE_TYPE/trained_models # change to your desired svaed model path
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE --saved_model_dir $SAVED_MODEL_DIR
done
