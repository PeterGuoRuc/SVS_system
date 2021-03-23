#!/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=2
stop_stage=2
ngpu=1
raw_data_dir=data
download_wavernn_vocoder=False

# model_name=conformer_full
model_name=glu
# model_name=rnn

expdir=exp/3_23_joint_3e-2

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
# set -o pipefail

./utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
  # Stage1: data preprocessing
  echo =============================
  echo " Stage1: data preprocessing "
  echo =============================
  mkdir -p ${raw_data_dir}
  python local/prepare_data.py data ..
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then 
  # Stage2: collect_stats
  echo =======================
  echo " Stage2: collect_stats "
  echo =======================

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train_joint.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/joint.yaml \
    --collect_stats True \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
  # Stage3: joint train
  echo ===============
  echo " Stage3: joint train"
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_train.log \
  train_joint.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/joint.yaml \
    --predictor_weight 0.03 \
    --initmodel_generator exp/3_17_glu_norm_pe_1e-4_Rcrop_semitone_pS/epoch_spec_loss_27.pth.tar \
    --initmodel_predictor exp/3_10_Dis_debug/epoch_loss_28.pth.tar \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi