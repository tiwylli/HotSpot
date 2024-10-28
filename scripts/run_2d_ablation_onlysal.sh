ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
# TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S") # Comment out if you don't want timestamp
IDENTIFIER='onlysal'                       # change to your desired identifier

CONFIG_DIR=$ROOT_DIR'configs/ablation_2d_onlysal.toml' # Change to your config file path
LOG_DIR=$ROOT_DIR'log/2d_ablation/'                    # Change to your log path
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for SHAPE_TYPE in 'house' 'target' 'bearing' 'fragments' 'boomerangs' 'seaurchin' 'snake' 'button' 'starhex' 'snowflake' 'square' 'L' 'circle' 'peace'; do
    cp -r scripts/$THIS_FILE $LOG_DIR
    cp -r $CONFIG_DIR $LOG_DIR
    echo $SHAPE_TYPE
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE
done
