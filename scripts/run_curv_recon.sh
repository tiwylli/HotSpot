ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")

CONFIG_DIR=$ROOT_DIR'configs/curv_recon.toml' # Change to your config file path
LOG_DIR=$ROOT_DIR'log/2d_curv/'                    # Change to your log path
IDENTIFIER='spin_curv'                       # change to your desired identifier
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for SHAPE_TYPE in 'boomerangs' 'fragments' 'house' 'seaurchin' 'target' 'circle' 'L' 'square' 'snowflake' 'starhex' 'button'  'bearing' 'snake' 'peace' ; do # for SHAPE_TYPE in 'circle' 'L' 'square' 'snowflake' 'starhex' 'button' 'target' 'bearing' 'snake' 'seaurchin' 'peace' 'boomerangs' 'fragments' 'house'; do
    echo "Run script for shape \"$SHAPE_TYPE\""
    SAVED_MODEL_DIR=$EXP_DIR/$SHAPE_TYPE/trained_models # Change to your desired saved model path, if evaluation is needed
    python3 train/train.py --config $CONFIG_DIR --log_dir $EXP_DIR/$SHAPE_TYPE --model_dir $MODEL_DIR --shape_type $SHAPE_TYPE --saved_model_dir $SAVED_MODEL_DIR
done
