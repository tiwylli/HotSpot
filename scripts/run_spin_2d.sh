ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
# TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S") # Comment out if you don't want timestamp

CONFIG_DIR=$ROOT_DIR'configs/curv_recon.toml' # Change to your config file path
LOG_DIR=$ROOT_DIR'out/2D/'                    # Change to your log path
IDENTIFIER='SPIN_debug'                       # change to your desired identifier
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for SHAPE_TYPE in 'circle'; do # for SHAPE_TYPE in 'circle' 'L' 'square' 'snowflake' 'starhex' 'button' 'target' 'bearing' 'snake' 'seaurchin' 'peace' 'boomerangs' 'fragments' 'house'; do
    cp -r scripts/$THIS_FILE $LOG_DIR
    cp -r $CONFIG_DIR $LOG_DIR
    echo $SHAPE_TYPE
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE
done
