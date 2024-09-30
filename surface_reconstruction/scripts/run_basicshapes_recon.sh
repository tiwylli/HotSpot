ROOT_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")

CONFIG_DIR=$ROOT_DIR'surface_reconstruction/configs/basic_shapes.toml' # change to your config file path
IDENTIFIER='2D_temp'                                                   # change to your desired identifier
LOG_DIR='./out/BasicShapes/'                                           # change to your desired log path
mkdir -p $LOG_DIR

for SHAPE_TYPE in 'L'; do # shapes: 'L', 'circle', 'snowflake', 'starAndHexagon'
    cp -r scripts/$THIS_FILE $LOG_DIR
    echo $SHAPE_TYPE
    python3 train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE
done
