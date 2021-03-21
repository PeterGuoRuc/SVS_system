#! /usr/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=0
stop_stage=0
ngpu=1

model_name=glu
# model_name=rnn

expdir=exp/3_19_filter
modeldir=exp/3_10_Dis_debug

set -e
set -u

./utils/parse_options.sh || exit 1;

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then 
  # Stage4: filter
  echo ===============
  echo " Stage4: filter "
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/filter.log \
  python svs_filter.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/filter.yaml \
    --prediction_path ${expdir} \
    --model_file ${modeldir}/epoch_loss_28.pth.tar \
    --stats_file ${modeldir}/feats_stats.npz \
    --stats_mel_file ${modeldir}/feats_mel_stats.npz

fi

