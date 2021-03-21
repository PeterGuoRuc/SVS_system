#! /usr/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=2
stop_stage=2
ngpu=1
raw_data_dir=data

expdir=exp/3_17_Dis_debug

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
  train_predictor.py \
    --db_joint True \
    --gpu_id 1 \
    -c conf/train_predictor.yaml \
    --collect_stats True \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz 
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then 
  # Stage3: train
  echo ===============
  echo " Stage3: train "
  echo ===============

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train_predictor.py \
    --db_joint True \
    --gpu_id 1 \
    -c conf/train_predictor.yaml \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then 
  # Stage4: inference
  echo ===============
  echo " Stage4: infer "
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/infer_predictor.log \
  infer_predictor.py \
    --db_joint True \
    --gpu_id 1 \
    -c conf/infer_predictor.yaml \
    --prediction_path ${expdir}/infer_result \
    --model_file ${expdir}/epoch_loss_28.pth.tar \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi

