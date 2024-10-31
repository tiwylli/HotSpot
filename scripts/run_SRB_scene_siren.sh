ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
IDENTIFIER='SIREN' # change to your desired identifier

CONFIG_DIR=$ROOT_DIR"configs/siren_shapenet.toml" # change to your config file path
DATASET_DIR=$ROOT_DIR'data/deep_geometric_prior_data/scans'    # change to your dataset path
LOG_DIR=$ROOT_DIR'log/3D/SRB/'                    # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

FOLDER_DIR=${DATASET_DIR}
for FILE_NAME in $FOLDER_DIR/*.ply; do # Iterate over all the files in the folder
    echo $FILE_NAME
    FILE_NAME=$(basename $FILE_NAME)
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR --train
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR --eval   
done


DATASET_DIR=$ROOT_DIR'data/scene_reconstruction'    # change to your dataset path
LOG_DIR=$ROOT_DIR'log/3D/scene/'                    # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

FOLDER_DIR=${DATASET_DIR}/
for FILE_NAME in $FOLDER_DIR/*.ply; do # Iterate over all the files in the folder
    echo $FILE_NAME
    FILE_NAME=$(basename $FILE_NAME)
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR --train
    python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR --eval   
done