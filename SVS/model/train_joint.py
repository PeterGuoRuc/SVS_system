"""Copyright [2020] [Jiatong Shi & Shuai Guo].

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# !/usr/bin/env python3

import logging
import numpy as np
import os

from SVS.model.network import RNN_Discriminator

from SVS.model.network import GLU_TransformerSVS
from SVS.model.network import GLU_TransformerSVS_combine

from SVS.model.network import LSTMSVS
from SVS.model.network import LSTMSVS_combine

from SVS.model.network import Joint_generator_predictor

from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset

from SVS.model.utils.gpu_util import use_single_gpu
from SVS.model.utils.loss import cal_psd2bark_dict
from SVS.model.utils.loss import cal_spread_function
from SVS.model.utils.loss import MaskedLoss
from SVS.model.utils.loss import PerceptualEntropy
from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset
from SVS.model.utils.transformer_optim import ScheduledOptim

from SVS.model.utils.utils import collect_stats
from SVS.model.utils.utils import save_model
from SVS.model.utils.utils import train_one_epoch_joint
from SVS.model.utils.utils import validate_one_epoch_joint

import sys
import time
import torch
from torch import nn


def count_parameters(model):
    """count_parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def Auto_save_model(
    args,
    epoch,
    model,
    optimizer,
    train_info,
    dev_info,
    logger,
    counter,
    epoch_to_save,
    save_loss_select="loss",
):
    """Auto_save_model."""
    if counter < args.num_saved_model:
        counter += 1
        # if dev_info[save_loss_select] in epoch_to_save.keys():
        #     counter -= 1
        #     continue
        epoch_to_save[dev_info[save_loss_select]] = epoch
        save_model(
            args,
            epoch,
            model,
            optimizer,
            train_info,
            dev_info,
            logger,
            save_loss_select,
        )

    else:
        sorted_dict_keys = sorted(epoch_to_save.keys(), reverse=True)
        select_loss = sorted_dict_keys[0]  # biggest spec_loss of saved models
        if dev_info[save_loss_select] < select_loss:
            epoch_to_save[dev_info[save_loss_select]] = epoch
            logging.info(f"### - {save_loss_select} - ###")
            logging.info(
                "add epoch: {:04d}, {}={:.4f}".format(
                    epoch, save_loss_select, dev_info[save_loss_select]
                )
            )

            if os.path.exists(
                "{}/epoch_{}_{}.pth.tar".format(
                    args.model_save_dir, save_loss_select, epoch_to_save[select_loss]
                )
            ):
                os.remove(
                    "{}/epoch_{}_{}.pth.tar".format(
                        args.model_save_dir,
                        save_loss_select,
                        epoch_to_save[select_loss],
                    )
                )
                logging.info(
                    "model of epoch:{} deleted".format(epoch_to_save[select_loss])
                )

            logging.info(
                "delete epoch: {:04d}, {}={:.4f}".format(
                    epoch_to_save[select_loss], save_loss_select, select_loss
                )
            )
            epoch_to_save.pop(select_loss)

            save_model(
                args,
                epoch,
                model,
                optimizer,
                train_info,
                dev_info,
                logger,
                save_loss_select,
            )

            logging.info(epoch_to_save)
    if len(sorted(epoch_to_save.keys())) > args.num_saved_model:
        raise ValueError("")

    return counter, epoch_to_save

def load_model_weights(model_load_dir, model, device):
    # Load model weights
    logging.info(f"Loading pretrained weights from {model_load_dir}")
    checkpoint = torch.load(model_load_dir, map_location=device)
    state_dict = checkpoint["state_dict"]
    model_dict = model.state_dict()
    state_dict_new = {}
    para_list = []

    for k, v in state_dict.items():
        # assert k in model_dict
        if (
            k == "normalizer.mean"
            or k == "normalizer.std"
            or k == "mel_normalizer.mean"
            or k == "mel_normalizer.std"
        ):
            continue
        if model_dict[k].size() == state_dict[k].size():
            state_dict_new[k] = v
        else:
            para_list.append(k)

    logging.info(
        f"Total {len(state_dict)} parameter sets, "
        f"loaded {len(state_dict_new)} parameter set"
    )

    if len(para_list) > 0:
        logging.warning(f"Not loading {para_list} because of different sizes")
    model.load_state_dict(state_dict_new)
    logging.info(f"Loaded checkpoint {model_load_dir}")
    model = model.to(device)

    return model

def train_joint(args):
    """train_joint."""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available() and args.auto_select_gpu is True:
        cvd = use_single_gpu()
        logging.info(f"GPU {cvd} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif torch.cuda.is_available() and args.auto_select_gpu is False:
        torch.cuda.set_device(args.gpu_id)
        logging.info(f"GPU {args.gpu_id} is used")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        device = torch.device("cpu")
        logging.info("Warning: CPU is used")

    train_set = SVSDataset(
        align_root_path=args.train_align,
        pitch_beat_root_path=args.train_pitch,
        wav_root_path=args.train_wav,
        char_max_len=args.char_max_len,
        max_len=args.num_frames,
        sr=args.sampling_rate,
        preemphasis=args.preemphasis,
        nfft=args.nfft,
        frame_shift=args.frame_shift,
        frame_length=args.frame_length,
        n_mels=args.n_mels,
        power=args.power,
        max_db=args.max_db,
        ref_db=args.ref_db,
        sing_quality=args.sing_quality,
        standard=args.standard,
        db_joint=args.db_joint,
        Hz2semitone=args.Hz2semitone,
        semitone_min=args.semitone_min,
        semitone_max=args.semitone_max,
        phone_shift_size=-1,
        semitone_shift=False,
    )

    dev_set = SVSDataset(
        align_root_path=args.val_align,
        pitch_beat_root_path=args.val_pitch,
        wav_root_path=args.val_wav,
        char_max_len=args.char_max_len,
        max_len=args.num_frames,
        sr=args.sampling_rate,
        preemphasis=args.preemphasis,
        nfft=args.nfft,
        frame_shift=args.frame_shift,
        frame_length=args.frame_length,
        n_mels=args.n_mels,
        power=args.power,
        max_db=args.max_db,
        ref_db=args.ref_db,
        sing_quality=args.sing_quality,
        standard=args.standard,
        db_joint=args.db_joint,
        Hz2semitone=args.Hz2semitone,
        semitone_min=args.semitone_min,
        semitone_max=args.semitone_max,
        phone_shift_size=-1,
        semitone_shift=False,
    )

    collate_fn_svs_train = SVSCollator(
        args.num_frames,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
        args.db_joint,
        args.random_crop,
        args.crop_min_length,
        args.Hz2semitone,
    )
    collate_fn_svs_val = SVSCollator(
        args.num_frames,
        args.char_max_len,
        args.use_asr_post,
        args.phone_size,
        args.n_mels,
        args.db_joint,
        False,  # random crop
        -1,  # crop_min_length
        args.Hz2semitone,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs_train,
        pin_memory=True,
    )
    dev_loader = torch.utils.data.DataLoader(
        dataset=dev_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs_val,
        pin_memory=True,
    )

    assert (
        args.feat_dim == dev_set[0]["spec"].shape[1]
        or args.feat_dim == dev_set[0]["mel"].shape[1]
    )

    if args.collect_stats:
        collect_stats(train_loader, args)
        logging.info("collect_stats finished !")
        quit()

    # init model_generate
    if args.model_type == "GLU_Transformer":
        if args.db_joint:
            model_generate = GLU_TransformerSVS_combine(
                phone_size=args.phone_size,
                singer_size=args.singer_size,
                embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                glu_num_layers=args.glu_num_layers,
                dropout=args.dropout,
                output_dim=args.feat_dim,
                dec_nhead=args.dec_nhead,
                dec_num_block=args.dec_num_block,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                local_gaussian=args.local_gaussian,
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
                device=device,
            )
        else:
            model_generate = GLU_TransformerSVS(
                phone_size=args.phone_size,
                embed_size=args.embedding_size,
                hidden_size=args.hidden_size,
                glu_num_layers=args.glu_num_layers,
                dropout=args.dropout,
                output_dim=args.feat_dim,
                dec_nhead=args.dec_nhead,
                dec_num_block=args.dec_num_block,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                local_gaussian=args.local_gaussian,
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
                device=device,
            )
    elif args.model_type == "LSTM":
        if args.db_joint:
            model_generate = LSTMSVS_combine(
                phone_size=args.phone_size,
                singer_size=args.singer_size,
                embed_size=args.embedding_size,
                d_model=args.hidden_size,
                num_layers=args.num_rnn_layers,
                dropout=args.dropout,
                d_output=args.feat_dim,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
                device=device,
                use_asr_post=args.use_asr_post,
            )
        else:
            model_generate = LSTMSVS(
                phone_size=args.phone_size,
                embed_size=args.embedding_size,
                d_model=args.hidden_size,
                num_layers=args.num_rnn_layers,
                dropout=args.dropout,
                d_output=args.feat_dim,
                n_mels=args.n_mels,
                double_mel_loss=args.double_mel_loss,
                Hz2semitone=args.Hz2semitone,
                semitone_size=args.semitone_size,
                device=device,
                use_asr_post=args.use_asr_post,
            )
    
    # init model_predict
    model_predict = RNN_Discriminator(
        embed_size=128,
        d_model=128,
        hidden_size=128,
        num_layers=2,
        n_specs=1025,
        singer_size=7,
        phone_size=43,
        simitone_size=59,
        dropout=0.1,
        bidirectional=True,
        device=device,
    )
    logging.info(f"*********** model_generate ***********")
    logging.info(f"{model_generate}")
    logging.info(f"The model has {count_parameters(model_generate):,} trainable parameters")

    logging.info(f"*********** model_predict ***********")
    logging.info(f"{model_predict}")
    logging.info(f"The model has {count_parameters(model_predict):,} trainable parameters")

    model_generate = load_model_weights(args.initmodel_generator, model_generate, device)
    model_predict = load_model_weights(args.initmodel_predictor, model_predict, device)

    model = Joint_generator_predictor(model_generate, model_predict)

    # setup optimizer
    if args.optimizer == "noam":
        optimizer = ScheduledOptim(
            torch.optim.Adam(
                model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
            ),
            args.hidden_size,
            args.noam_warmup_steps,
            args.noam_scale,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09
        )
        if args.scheduler == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=args.lr,
                steps_per_epoch=len(train_loader),
                epochs=args.max_epochs,
            )
        elif args.scheduler == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", verbose=True, patience=10, factor=0.5
            )
        elif args.scheduler == "ExponentialLR":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, verbose=True, gamma=0.9886
            )
    else:
        raise ValueError("Not Support Optimizer")

    # setup loss function
    loss_predict = nn.CrossEntropyLoss(reduction="sum")

    if args.loss == "l1":
        loss_generate = MaskedLoss("l1", mask_free=args.mask_free)
    elif args.loss == "mse":
        loss_generate = MaskedLoss("mse", mask_free=args.mask_free)
    else:
        raise ValueError("Not Support Loss Type")

    if args.perceptual_loss > 0:
        win_length = int(args.sampling_rate * args.frame_length)
        psd_dict, bark_num = cal_psd2bark_dict(
            fs=args.sampling_rate, win_len=win_length
        )
        sf = cal_spread_function(bark_num)
        loss_perceptual_entropy = PerceptualEntropy(
            bark_num, sf, args.sampling_rate, win_length, psd_dict
        )
    else:
        loss_perceptual_entropy = None


    # Training
    generator_loss_epoch_to_save = {}
    generator_loss_counter = 0
    spec_loss_epoch_to_save = {}
    spec_loss_counter = 0
    predictor_loss_epoch_to_save = {}
    predictor_loss_counter = 0

    for epoch in range(0, 1 + args.max_epochs):
        """Train Stage"""
        start_t_train = time.time()
        train_info = train_one_epoch_joint(
            train_loader,
            model,
            device,
            optimizer,
            loss_generate,
            loss_predict,
            loss_perceptual_entropy,
            epoch,
            args,
        )
        end_t_train = time.time()

        # Print Total info
        out_log = "Train epoch: {:04d} ".format(epoch)
        if args.optimizer == "noam":
            out_log += "lr: {:.6f}, \n\t".format(optimizer._optimizer.param_groups[0]["lr"])
        elif args.optimizer == "adam":
            out_log += "lr: {:.6f}, \n\t".format(optimizer.param_groups[0]["lr"])

        out_log += "total_loss: {:.4f} \n\t".format(train_info["loss"])

        # Print Generator info
        if args.vocoder_category == "wavernn":
            out_log += "generator_loss: {:.4f} ".format(train_info["generator_loss"])
        else:
            out_log += "generator_loss: {:.4f}, spec_loss: {:.4f} ".format(
                train_info["generator_loss"], train_info["spec_loss"]
            )
        if args.n_mels > 0:
            out_log += "mel_loss: {:.4f}, ".format(train_info["mel_loss"])
        if args.perceptual_loss > 0:
            out_log += "pe_loss: {:.4f}\n\t".format(train_info["pe_loss"])

        # Print Predictor info
        out_log += "predictor_loss: {:.4f}, singer_loss: {:.4f}, ".format(
            train_info["predictor_loss"],
            train_info["singer_loss"],
        )
        out_log += "phone_loss: {:.4f}, semitone_loss: {:.4f} \n\t\t".format(
            train_info["phone_loss"],
            train_info["semitone_loss"],
        )
        out_log += "singer_accuracy: {:.4f}%, ".format(
            train_info["singer_accuracy"] * 100,
        )
        out_log += "phone_accuracy: {:.4f}%, semitone_accuracy: {:.4f}% ".format(
            train_info["phone_accuracy"] * 100,
            train_info["semitone_accuracy"] * 100,
        )

        logging.info("{} time: {:.2f}s".format(out_log, end_t_train - start_t_train))

        """Dev Stage"""
        torch.backends.cudnn.enabled = False  # 莫名的bug，关掉才可以跑

        # start_t_dev = time.time()
        dev_info = validate_one_epoch_joint(
            dev_loader,
            model,
            device,
            optimizer,
            loss_generate,
            loss_predict,
            loss_perceptual_entropy,
            epoch,
            args,
        )
        end_t_dev = time.time()

        # Print Total info
        dev_log = "Dev epoch: {:04d} ".format(epoch)
        if args.optimizer == "noam":
            dev_log += "lr: {:.6f}, \n\t".format(optimizer._optimizer.param_groups[0]["lr"])
        elif args.optimizer == "adam":
            dev_log += "lr: {:.6f}, \n\t".format(optimizer.param_groups[0]["lr"])

        dev_log += "total_loss: {:.4f} \n\t".format(dev_info["loss"])

        # Print Generator info
        if args.vocoder_category == "wavernn":
            dev_log += "generator_loss: {:.4f} ".format(dev_info["generator_loss"])
        else:
            dev_log += "generator_loss: {:.4f}, spec_loss: {:.4f} ".format(
                dev_info["generator_loss"], dev_info["spec_loss"]
            )
        if args.n_mels > 0:
            dev_log += "mel_loss: {:.4f}, ".format(dev_info["mel_loss"])
        if args.perceptual_loss > 0:
            dev_log += "pe_loss: {:.4f}\n\t".format(dev_info["pe_loss"])

        # Print Predictor info
        dev_log += "predictor_loss: {:.4f}, singer_loss: {:.4f}, ".format(
            dev_info["predictor_loss"],
            dev_info["singer_loss"],
        )
        dev_log += "phone_loss: {:.4f}, semitone_loss: {:.4f} \n\t\t".format(
            dev_info["phone_loss"],
            dev_info["semitone_loss"],
        )
        dev_log += "singer_accuracy: {:.4f}%, ".format(
            dev_info["singer_accuracy"] * 100,
        )
        dev_log += "phone_accuracy: {:.4f}%, semitone_accuracy: {:.4f}% ".format(
            dev_info["phone_accuracy"] * 100,
            dev_info["semitone_accuracy"] * 100,
        )
        logging.info("{} time: {:.2f}s".format(dev_log, end_t_dev - start_t_train))

        sys.stdout.flush()

        torch.backends.cudnn.enabled = True

        """Save model Stage"""
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        (generator_loss_counter, generator_loss_epoch_to_save) = Auto_save_model(
            args,
            epoch,
            model,
            optimizer,
            train_info,
            dev_info,
            None,  # logger
            generator_loss_counter,
            generator_loss_epoch_to_save,
            save_loss_select="generator_loss",
        )

        (spec_loss_counter, spec_loss_epoch_to_save) = Auto_save_model(
            args,
            epoch,
            model,
            optimizer,
            train_info,
            dev_info,
            None,  # logger
            spec_loss_counter,
            spec_loss_epoch_to_save,
            save_loss_select="spec_loss",
        )

        (predictor_loss_counter, predictor_loss_epoch_to_save) = Auto_save_model(
            args,
            epoch,
            model,
            optimizer,
            train_info,
            dev_info,
            None,  # logger
            predictor_loss_counter,
            predictor_loss_epoch_to_save,
            save_loss_select="predictor_loss",
        )

        