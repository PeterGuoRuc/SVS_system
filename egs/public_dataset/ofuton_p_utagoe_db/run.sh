#!/bin/bash

# Copyright 2020 RUC & Johns Hopkins University (author: Shuai Guo, Jiatong Shi, Lei Li)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=0
stop_stage=100
ngpu=1
raw_data_dir=downloads
expdir=exp/rnn
download_wavernn_vocoder=True
vocoder=wavernn

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


./utils/parse_options.sh || exit 1;


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  # Stage0: download data
  echo =======================
  echo " Stage0: download data "
  echo =======================
  mkdir -p ${raw_data_dir}
  ./local/download_and_untar.sh ${raw_data_dir} 'https://doc-00-bs-docs.googleusercontent.com/docs/securesc/f7j5086anjlk94elkcocfjja7egds00v/onaktddvgna79e13m1g8p2p8rg1hfnpm/1611905625000/00948940906475156563/00948940906475156563/136Ir-35Fe9fGlwGKP_ETBKvAy-bTpYIz?e=download&authuser=0&nonce=5uk3dtcl40grk&user=00948940906475156563&hash=c8sj8n26414b' OFUTON_P_UTAGOE_DB.zip
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Stage1: data preprocessing & format into different set(trn/val/tst)
  echo ============================
  echo " Stage1: data preprocessing "
  echo ============================

  if [ ${download_wavernn_vocoder} = True ]; then
    wget -nc https://raw.githubusercontent.com/pppku/model_zoo/main/wavernn/latest_weights.pyt -P ${expdir}/model/wavernn
    python local/prepare_data.py ${raw_data_dir}/OFUTON_P_UTAGOE_DB ${raw_data_dir}/OFUTON_P_UTAGOE_DB data \
      --label_type r --wav_extention wav \
      --window_size 50 \
      --shift_size 12.5 \
      --sil pau sil
  else
      python local/prepare_data.py ${raw_data_dir}/OFUTON_P_UTAGOE_DB \
        ${raw_data_dir}/OFUTON_P_UTAGOE_DB data \
          --label_type ns \
          --sil pau sil
  fi
  ./local/train_dev_test_split.sh data train dev test

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  # Stage2: collect_stats
  echo =======================
  echo " Stage2: collect_stats "
  echo =======================

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train.py \
    -c conf/train_rnn_wavernn.yaml \
    --collect_stats True \
    --model_save_dir ${expdir} \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  # Stage3: train
  echo ===============
  echo " Stage3: train "
  echo ===============

  if [ ${download_wavernn_vocoder} = True ]; then
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
    train.py \
      -c conf/train_rnn_wavernn.yaml \
      --gpu_id -1 \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz \
      --vocoder_category ${vocoder} \
      --wavernn_voc_model ${expdir}/model/wavernn/latest_weights.pyt
  else
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
    train.py \
      -c conf/train_rnn_norm_perp.yaml \
      --gpu_id -1 \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
  fi

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  # Stage4: inference
  echo ===============
  echo " Stage4: infer "
  echo ===============

  if [ ${download_wavernn_vocoder} = True ]; then
    ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer.log \
    infer.py -c conf/infer_rnn_wavernn.yaml \
      --prediction_path ${expdir}/infer_result \
      --model_file ${expdir}/epoch_loss_102.pth.tar \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz \
      --vocoder_category ${vocoder} \
      --wavernn_voc_model ${expdir}/model/wavernn/latest_weights.pyt
  else
    ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer.log \
    infer.py -c conf/infer_rnn_norm_perp.yaml \
      --prediction_path ${expdir}/infer_result \
      --model_file ${expdir}/epoch_spec_loss_117.pth.tar \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
  fi

fi

