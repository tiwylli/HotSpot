ROOT_DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S")

CONFIG_DIR=$ROOT_DIR'surface_reconstruction/configs/shapenet.toml' # change to your config file path
IDENTIFIER='SPIN_temp'                                             # change to your desired identifier
DATASET_DIR=$ROOT_DIR'data/NSP_dataset/'                           # change to your dataset path
LOG_DIR='./out/ShapeNet/'                                          # change to your desired log path
mkdir -p $LOG_DIR

for SHAPE_NAME in 'lamp'; do # use the shapes you want in the dataset
    FOLDER_DIR=${DATASET_DIR}/$SHAPE_NAME/
    cp -r scripts/$THIS_FILE $LOG_DIR
    echo $SHAPE_NAME
    for FILE_NAME in 'd1aed86c38d9ea6761462fc0fa9b0bb4.ply'; do # use the scans you want
        echo $FILE_NAME
        python3 train.py --config $CONFIG_DIR --log_dir $LOG_DIR/$IDENTIFIER$TIMESTAMP/$SHAPE_NAME --data_dir $FOLDER_DIR --file_name $FILE_NAME --model_dir $MODEL_DIR
    done
done
