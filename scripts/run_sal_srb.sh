ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")
IDENTIFIER='SAL' # change to your desired identifier

CONFIG_DIR=$ROOT_DIR"configs/sal_shapenet.toml" # change to your config file path
DATASET_DIR=$ROOT_DIR'data/deep_geometric_prior_data/scans'    # change to your dataset path
LOG_DIR=$ROOT_DIR'log/3D/SRB/'                    # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory

for FILENAME in 'gargoyle.ply' 'daratech.ply' 'lord_quas.ply' 'anchor.ply' 'dc.ply'; do # use the shapes you want in the dataset
    FOLDER_DIR=${DATASET_DIR}/$SHAPE_NAME/
    echo $SHAPE_NAME
    # for FILE_NAME in 'd1b15263933da857784a45ea6efa1d77.ply'; do # use the scans you want
    # iterate over all the files in the folder
    for FILE_NAME in $FOLDER_DIR/*.ply; do
        echo $FILE_NAME
        FILE_NAME=$(basename $FILE_NAME)
        python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR
    done
done
