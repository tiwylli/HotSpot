ROOT_DIR=$(dirname $(dirname "$(readlink -f "$0")"))'/'
MODEL_DIR=$ROOT_DIR'models'
THIS_FILE=$(basename "$0")
TIMESTAMP=$(date +"-%Y-%m-%d-%H-%M-%S") # Comment out if you don't want timestamp

CONFIG_DIR=$ROOT_DIR'configs/diff_2d_eik_heat.toml' # Change to your config file path
LOG_DIR=$ROOT_DIR'log/2d_different/'            # Change to your log path
IDENTIFIER='2d_different_comparison'                        # change to your desired identifier
EXP_DIR=$LOG_DIR$IDENTIFIER$TIMESTAMP/
mkdir -p $EXP_DIR
cp -r scripts/$THIS_FILE $EXP_DIR # Copy this script to the experiment directory
cp -r $CONFIG_DIR $EXP_DIR        # Copy the config file to the experiment directory


for SHAPE_TYPE in 'fragments' 'house'; do # 
    for eik_coe in 0.0 0.01 0.05 0.1 0.5 1.0; do # 0.0 0.01 0.05 0.1 0.5 1.0; do
        for heat_lambda in 100 1 0.5 5 50 10 ; do #100 1 0.5 5 50 10; do
            echo "Run script for shape \"$SHAPE_TYPE\" with heat_lambda=$heat_lambda and eik_coe=$eik_coe"
            
            # Define log directory and saved model directory with parameters
            LOG_DIR=$EXP_DIR/$SHAPE_TYPE/heat_lambda_${heat_lambda}_eik_coe_${eik_coe}
            SAVED_MODEL_DIR=$LOG_DIR/trained_models # Save models in the log directory

            # Run the training script with updated log and saved model paths
            python3 train/train.py --config $CONFIG_DIR --log_dir $LOG_DIR --model_dir $MODEL_DIR \
            --shape_type $SHAPE_TYPE --heat_lambda $heat_lambda --eikonal_decay_params $eik_coe $eik_coe \
            --saved_model_dir $SAVED_MODEL_DIR
        done
    done
done
