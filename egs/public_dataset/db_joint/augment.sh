#! /usr/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=0
stop_stage=0
ngpu=1

model_name=glu
# model_name=rnn

expdir=exp/3_18_augment
modeldir=exp/3_17_glu_norm_pe_1e-4_Rcrop_semitone_pS

set -e
set -u

./utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
  # Stage4: augment
  echo ===============
  echo " Stage4: augment "
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/augment.log \
  python svs_augment.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/augment_${model_name}.yaml \
    --prediction_path ${expdir}/train_result \
    --model_file ${modeldir}/epoch_spec_loss_27.pth.tar \
    --stats_file ${modeldir}/feats_stats.npz \
    --stats_mel_file ${modeldir}/feats_mel_stats.npz

fi

