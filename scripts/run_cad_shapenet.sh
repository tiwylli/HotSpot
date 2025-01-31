ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")
IDENTIFIER='CAD-lamp-all' # change to your desired identifier

CONFIG_DIR=$ROOT_DIR"configs/cad_shapenet.toml" # change to your config file path
DATASET_DIR=$ROOT_DIR'data/NSP_dataset/'         # change to your dataset path
LOG_DIR=$ROOT_DIR'log/3D/ShapeNet/'              # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for SHAPE_NAME in 'lamp'; do # use the shapes you want in the dataset
    FOLDER_DIR=${DATASET_DIR}/$SHAPE_NAME/
    echo $SHAPE_NAME
    for FILE_NAME in $FOLDER_DIR/*.ply; do # Iterate over all the files in the folder
    # for FILE_NAME in $FOLDER_DIR/cbc946b4f4c022305e524bb779a94481.ply; do # Iterate over all the files in the folder
    # for FILE_NAME in $FOLDER_DIR/d1b15263933da857784a45ea6efa1d77.ply; do # Iterate over all the files in the folder
    # for FILE_NAME in $FOLDER_DIR/d284b73d5983b60f51f77a6d7299806.ply; do # Iterate over all the files in the folder
        echo $FILE_NAME
        FILE_NAME=$(basename $FILE_NAME)
        python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR
    done
done
