#!/bin/bash
DIR=$(dirname $(dirname $(dirname "$(readlink -f "$0")")))  # Should point to your DiGS path
echo "If $DIR is not the correct path for your repository, set it manually at the variable DIR"
cd $DIR/sanitychecks/ # To call python scripts correctly
OUTDIR='./out/'
IDENTIFIER='heat_lambda8_smallNet_temp'
LOGDIRNAME=${OUTDIR}${IDENTIFIER}'/'
LOGDIR=${LOGDIRNAME}'log/' #change to your desired log directory
mkdir -p $LOGDIR
FILE=`basename "$0"`
cp scripts/$FILE $LOGDIR # saves copy of this script so you know the args used


### MODEL HYPER-PARAMETERS ###
##############################
LAYERS=4
DECODER_HIDDEN_DIM=128
NL='relu' # 'sine' | 'relu' | 'softplus'
SPHERE_INIT_PARAMS=(1.6 1.0)
INIT_TYPE='geometric_relu' #siren | geometric_sine | geometric_relu | mfgi
NEURON_TYPE='linear' #linear | quadratic
### LOSS HYPER-PARAMETERS ###
#############################
LOSS_TYPE='igr_wo_eik_w_heat' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
# LOSS_TYPE='siren_wo_n_w_div' # 'siren_wo_n_w_div' | 'siren_wo_n' | 'siren_w_div' | 'siren'
LOSS_WEIGHTS=(3e3 1e2 1e2 5e1 1e2 0 5e2) # sdf, inter, normal, eikonal, div, latent, heat
DIV_TYPE='dir_l1' # 'dir_l1' | 'dir_l2' | 'full_l1' | 'full_l2'
DIVDECAY='linear' # 'linear' | 'quintic' | 'step'
DECAY_PARAMS=(1e2 0.2 1e2 0.4 0.0 0.0)
### DOMAIN HYPER-PARAMETERS ###
###############################
GRID_RES=256
NONMNFLD_SAMPLE_TYPE='grid'
# N_RANDOM_SAMPLES=2e16
N_RANDOM_SAMPLES=4096
NPOINTS=15000
GRID_RANGE=1.2
### TRAINING HYPER-PARAMETERS ###
#################################
NSAMPLES=1000
BATCH_SIZE=1
GPU=0
NEPOCHS=1
EVALUATION_EPOCH=0
LR=5e-5
GRAD_CLIP_NORM=10.0
### TESTING ARGUMENTS ###
#################################
EPOCHS_N_EVAL=($(seq 0 100 9900)) # use this to generate images of different iterations
# EPOCHS_N_EVAL=($(seq 0 100 4900)) # use this to generate images of different iterations

HEAT_LAMBDA=16
NONMNFLD_SAMPLE_STD2=0.09

# for SHAPE in 'starAndHexagon'
for SHAPE in 'L'
# for SHAPE in 'L' 'circle' 'snowflake'
do
  LOGDIR=${LOGDIRNAME}${NONMNFLD_SAMPLE_TYPE}'_sampling_'${GRID_RES}'/'${SHAPE}'/'
  python3 train_basic_shape.py --logdir $LOGDIR --shape_type $SHAPE --grid_res $GRID_RES --loss_type $LOSS_TYPE --inter_loss_type 'exp' --num_epochs $NEPOCHS --gpu_idx $GPU --n_samples $NSAMPLES --n_points $NPOINTS --batch_size $BATCH_SIZE --lr ${LR} --nonmnfld_sample_type $NONMNFLD_SAMPLE_TYPE --decoder_n_hidden_layers $LAYERS  --decoder_hidden_dim $DECODER_HIDDEN_DIM --div_decay $DIVDECAY --div_decay_params ${DECAY_PARAMS[@]} --div_type $DIV_TYPE --init_type ${INIT_TYPE} --neuron_type ${NEURON_TYPE} --nl ${NL} --sphere_init_params ${SPHERE_INIT_PARAMS[@]} --loss_weights ${LOSS_WEIGHTS[@]} --grad_clip_norm ${GRAD_CLIP_NORM[@]} --heat_lambda ${HEAT_LAMBDA} --nonmnfld_sample_std2 ${NONMNFLD_SAMPLE_STD2} --n_random_samples ${N_RANDOM_SAMPLES} --grid_range ${GRID_RANGE}
  python3 test_basic_shape.py --logdir $LOGDIR --gpu_idx $GPU --epoch_n "${EPOCHS_N_EVAL[@]}"
done
