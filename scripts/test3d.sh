ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")
# IDENTIFIER='SPIN-StEik-failures' # change to your desired identifier
IDENTIFIER='3d-test' # change to your desired identifier
CONFIG_DIR=$ROOT_DIR"configs/test3d.toml" # change to your config file path
DATASET_DIR=$ROOT_DIR'data/NSP_dataset/'         # change to your dataset path
LOG_DIR=$ROOT_DIR'log/3d_shapeNet/'              # change to your desired log path
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory
#for SHAPE_NAME in 'lamp' 'rifle' 'sofa' 'table' 'telephone' 'watercraft' ; do # use the shapes you want in the dataset
#for SHAPE_NAME in 'loudspeaker' 'airplane' 'bench' 'cabinet' 'car' 'chair' 'display' ; do
# for SHAPE_NAME in 'lamp' 'bench' 'cabinet' 'loudspeaker' 'car' 'chair' 'display' 'airplane' 'rifle' 'sofa' 'table' 'telephone' 'watercraft' ; do # use the shapes you want in the dataset
# for SHAPE_NAME in 'lamp'; do # use the shapes you want in the dataset
for SHAPE_NAME in 'loudspeaker'; do # use the shapes you want in the dataset
    FOLDER_DIR=${DATASET_DIR}/$SHAPE_NAME/
    echo $SHAPE_NAME
    #for FILE_NAME in 'd16bb6b2f26084556acbef8d3bef8f28.ply' 'd284b73d5983b60f51f77a6d7299806.ply' 'd1b15263933da857784a45ea6efa1d77.ply' 'd217e8ab61670bbb433009863c91a425.ply'; do # use the scans you want
    # for FILE_NAME in 'd16bb6b2f26084556acbef8d3bef8f28.ply' 'd284b73d5983b60f51f77a6d7299806.ply' 'd1b15263933da857784a45ea6efa1d77.ply' 'd217e8ab61670bbb433009863c91a425.ply'; do # use the scans you want
    #for FILE_NAME in 'c9de3e18847044da47e2162b6089a53e.ply'; do # use the scans you want
    for FILE_NAME in 'cb3bc7b6610acb7d7f38a9bfed62447a.ply'; do # use the scans you want
    #for FILE_NAME in $FOLDER_DIR/*.ply; do # Iterate over all the files in the folder
        echo $FILE_NAME
        FILE_NAME=$(basename $FILE_NAME)
        python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR
    done
done