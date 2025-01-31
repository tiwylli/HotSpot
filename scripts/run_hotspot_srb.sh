ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")
IDENTIFIER='SPIN-all' # change to your desired identifier

CONFIG_DIR=$ROOT_DIR"configs/hotspot_srb.toml"         # change to your config file path
DATASET_DIR=$ROOT_DIR'data/deep_geometric_prior_data/' # change to your dataset path
LOG_DIR=$ROOT_DIR'log/3D/SRB/'                         # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for SHAPE_NAME in 'anchor' 'daratech' 'dc' 'gargoyle' 'lord_quas'; do # use the shapes you want in the dataset
    FOLDER_DIR=${DATASET_DIR}/scans
    echo $SHAPE_NAME
    FILE_NAME=$SHAPE_NAME.ply
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR
done
