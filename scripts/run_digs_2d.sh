ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")
IDENTIFIER='DiGS' # change to your desired identifier

CONFIG_DIR=$ROOT_DIR'configs/digs_2d.toml' # change to your config file path
LOG_DIR=$ROOT_DIR'log/2d_curv/'                 # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for SHAPE_TYPE in 'snowflake' 'circle' 'L' 'square' 'starhex' 'button' 'target' 'bearing' 'snake' 'seaurchin' 'peace' 'boomerangs' 'fragments' 'house'; do
    # for SHAPE_TYPE in 'L'; do
    echo $SHAPE_TYPE
    python3 train/train.py --config $CONFIG_DIR --log_dir $EXP_DIR/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE
done
