#!/bin/bash

# Copyright 2020 RUC (author: Shuai Guo)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


stage=5
stop_stage=5
ngpu=1
raw_data_dir=data
download_wavernn_vocoder=False

# model_name=conformer_full
model_name=glu
# model_name=rnn

expdir=exp/3_20_glu_filter_debug

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

  if [ ${download_wavernn_vocoder} = True ]; then
    wget -nc https://raw.githubusercontent.com/pppku/model_zoo/main/wavernn/latest_weights.pyt -P ${expdir}/model/wavernn
  fi

fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # Stage2: collect_stats
  echo =======================
  echo " Stage2: collect_stats "
  echo =======================

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats.log \
  train.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/train_${model_name}.yaml \
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

  if [ ${download_wavernn_vocoder} = True ]; then
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
    train.py \
      --db_joint True \
      --gpu_id 0 \
      -c conf/train_${model_name}_wavernn.yaml \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz \
      --vocoder_category ${vocoder} \
      --wavernn_voc_model ${expdir}/model/wavernn/latest_weights.pyt
  else
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/svs_train.log \
    train.py \
      --db_joint True \
      --gpu_id 0 \
      -c conf/train_${model_name}.yaml \
      --model_save_dir ${expdir} \
      --stats_file ${expdir}/feats_stats.npz \
      --stats_mel_file ${expdir}/feats_mel_stats.npz
  fi

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then 
  # Stage3: train
  echo ===============
  echo " Stage3: train with augmented data"
  echo ===============

  ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats_filter_aug.log \
  train.py \
    --db_joint True \
    --gpu_id 1 \
    -c conf/train_${model_name}.yaml \
    --model_save_dir ${expdir}/filter_aug \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz \
    --filter_wav_path /data1/gs/SVS_system/egs/public_dataset/db_joint/local/filter_wav_filename.txt \
    --filter_weight 0.1
    # --initmodel ${expdir}/epoch_spec_loss_27.pth.tar \
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then 
  # Stage4: inference
  echo ===============
  echo " Stage4: infer "
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer.log \
  infer.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/infer_${model_name}.yaml \
    --prediction_path ${expdir}/infer_result \
    --model_file ${expdir}/epoch_spec_loss_41.pth.tar \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then 
  # Stage4: inference
  echo ===============
  echo " Stage4: infer augmented model"
  echo ===============

  ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer_aug.log \
  infer.py \
    --db_joint True \
    --gpu_id 0 \
    -c conf/infer_${model_name}.yaml \
    --prediction_path ${expdir}/infer_result_aug \
    --model_file ${expdir}/filter_aug/epoch_spec_loss_17.pth.tar \
    --stats_file ${expdir}/feats_stats.npz \
    --stats_mel_file ${expdir}/feats_mel_stats.npz

fi

# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then 
#   # Stage5: train
#   echo =============================
#   echo " Stage5: kiritan fine-tune "
#   echo =============================

#   ${cuda_cmd} --gpu ${ngpu} ${expdir}/stats_finetune.log \
#   train.py \
#     --finetune_dbname kiritan \
#     --initmodel ${expdir}/epoch_spec_loss_27.pth.tar \
#     --db_joint True \
#     --gpu_id 0 \
#     -c conf/train_${model_name}.yaml \
#     --model_save_dir ${expdir}/fintune \
#     --stats_file ${expdir}/feats_stats.npz \
#     --stats_mel_file ${expdir}/feats_mel_stats.npz

# fi

# if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then 
#   # Stage6: fine-tune infer
#   echo ===================================
#   echo " Stage6: kiritan fine-tune infer "
#   echo ===================================

#   ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer_kiritan.log \
#   infer.py \
#     --finetune_dbname kiritan \
#     --db_joint True \
#     --gpu_id 0 \
#     -c conf/infer_${model_name}.yaml \
#     --prediction_path ${expdir}/infer_result_fintune \
#     --model_file ${expdir}/fintune/epoch_spec_loss_10.pth.tar \
#     --stats_file ${expdir}/feats_stats.npz \
#     --stats_mel_file ${expdir}/feats_mel_stats.npz

# fi

# if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then 
#   # Stage6: fine-tune infer
#   echo ===================================
#   echo " Stage6: kiritan 7-combine infer "
#   echo ===================================

#   ${cuda_cmd} -gpu ${ngpu} ${expdir}/svs_infer_kiritan_direct.log \
#   infer.py \
#     --finetune_dbname kiritan \
#     --db_joint True \
#     --gpu_id 0 \
#     -c conf/infer_${model_name}.yaml \
#     --prediction_path ${expdir}/infer_result_direct \
#     --model_file ${expdir}/epoch_spec_loss_27.pth.tar \
#     --stats_file ${expdir}/feats_stats.npz \
#     --stats_mel_file ${expdir}/feats_mel_stats.npz

# fi