import logging
import numpy as np
import os
import yamlargparse
from SVS.model.layers.global_mvn import GlobalMVN
from SVS.model.network import RNN_Discriminator
from SVS.model.utils.loss import MaskedLoss

from SVS.model.utils.SVSDataset import SVSCollator
from SVS.model.utils.SVSDataset import SVSDataset_filter

from SVS.model.utils.utils import AverageMeter
from SVS.model.utils.utils import log_figure
from SVS.model.utils.utils import log_mel
from SVS.model.utils.utils import spectrogram2wav
from SVS.model.utils.utils import griffin_lim

import SVS.utils.metrics as Metrics
import time
import torch

import librosa
from scipy import signal
import soundfile as sf


def parse_args():
    parser = yamlargparse.ArgumentParser(description="SVS training")
    parser.add_argument(
        "-c",
        "--config",
        help="config file path",
        action=yamlargparse.ActionConfigFile,
    )
    parser.add_argument("--test_align", help="alignment data dir used for validation.")
    parser.add_argument("--test_pitch", help="pitch data dir used for validation.")
    parser.add_argument("--test_wav", help="wave data dir used for validation")
    parser.add_argument("--model_file", help="model file for prediction.")
    parser.add_argument(
        "--prediction_path", help="prediction result output (e.g. wav, png)."
    )
    parser.add_argument(
        "--model_type",
        default="GLU_Transformer",
        help="Type of model (New_Transformer or GLU_Transformer or LSTM)",
    )
    parser.add_argument(
        "--num_frames",
        default=500,
        type=int,
        help="number of frames in one utterance",
    )
    parser.add_argument(
        "--db_joint",
        type=bool,
        default=False,
        help="Combine multiple datasets & add singer embedding",
    )
    parser.add_argument(
        "--Hz2semitone",
        type=bool,
        default=False,
        help="Transfer f0 value into semitone",
    )
    parser.add_argument(
        "--semitone_size",
        type=int,
        default=59,
        help="Semitone size of your dataset, can be found in data/semitone_set.txt",
    )
    parser.add_argument(
        "--semitone_min",
        type=str,
        default="F_1",
        help="Minimum semitone of your dataset, can be found in data/semitone_set.txt",
    )
    parser.add_argument(
        "--semitone_max",
        type=str,
        default="D_6",
        help="Maximum semitone of your dataset, can be found in data/semitone_set.txt",
    )
    parser.add_argument(
        "--char_max_len", default=100, type=int, help="max length for character"
    )
    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of cpu workers"
    )
    parser.add_argument(
        "--decode_sample", default=-1, type=int, help="samples to decode"
    )
    parser.add_argument("--frame_length", default=0.06, type=float)
    parser.add_argument("--frame_shift", default=0.03, type=float)
    parser.add_argument("--sampling_rate", default=44100, type=int)
    parser.add_argument("--preemphasis", default=0.97, type=float)
    parser.add_argument("--n_mels", default=80, type=int)
    parser.add_argument("--power", default=1.2, type=float)
    parser.add_argument("--max_db", default=100, type=int)
    parser.add_argument("--ref_db", default=20, type=int)
    parser.add_argument("--nfft", default=2048, type=int)
    parser.add_argument("--phone_size", default=67, type=int)
    parser.add_argument("--singer_size", default=10, type=int)
    parser.add_argument("--feat_dim", default=1324, type=int)
    parser.add_argument("--embedding_size", default=256, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument(
        "--glu_num_layers", default=1, type=int, help="number of glu layers"
    )
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--dec_num_block", default=6, type=int)
    parser.add_argument("--num_rnn_layers", default=2, type=int)
    parser.add_argument("--dec_nhead", default=4, type=int)
    parser.add_argument("--local_gaussian", default=False, type=bool)
    parser.add_argument("--seed", default=666, type=int)
    parser.add_argument(
        "--use_tfb",
        dest="use_tfboard",
        help="whether use tensorboard",
        action="store_true",
    )
    parser.add_argument("--loss", default="l1", type=str)
    parser.add_argument("--perceptual_loss", default=-1, type=float)
    parser.add_argument("--use_pos_enc", default=0, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--use_asr_post", default=False, type=bool)
    parser.add_argument("--sing_quality", default="conf/sing_quality.csv", type=str)
    parser.add_argument("--standard", default=-1, type=int)

    parser.add_argument("--stats_file", default="", type=str)
    parser.add_argument("--stats_mel_file", default="", type=str)
    parser.add_argument("--collect_stats", default=False, type=bool)
    parser.add_argument("--normalize", default=False, type=bool)
    parser.add_argument("--num_saved_model", default=5, type=int)

    parser.add_argument("--accumulation_steps", default=1, type=int)
    parser.add_argument("--auto_select_gpu", default=True, type=bool)
    parser.add_argument("--gpu_id", default=1, type=int)
    parser.add_argument("--double_mel_loss", default=False, type=float)
    parser.add_argument("--vocoder_category", default="griffin", type=str)
    
    args = parser.parse_args()
    return args

def count_parameters(model):
    """count_parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(args):
    torch.cuda.set_device(args.gpu_id)
    logging.info(f"GPU {args.gpu_id} is used")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # prepare model
    model = RNN_Discriminator(
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
    logging.info(f"{model}")
    logging.info(f"The model has {count_parameters(model):,} trainable parameters")

    # Load model weights
    logging.info(f"Loading pretrained weights from {args.model_file}")
    checkpoint = torch.load(args.model_file, map_location=device)
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
    logging.info(f"Loaded checkpoint {args.model_file}")
    model = model.to(device)
    model.eval()

    return model, device


def data_filter(args):

    model, device = load_model(args)

    start_t_test = time.time()

    # Decode
    test_set = SVSDataset_filter(
        align_root_path=args.test_align,
        pitch_beat_root_path=args.test_pitch,
        wav_root_path=args.test_wav,
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
        standard=args.standard,
        sing_quality=args.sing_quality,
        Hz2semitone=args.Hz2semitone,
        semitone_min=args.semitone_min,
        semitone_max=args.semitone_max,
    )
    collate_fn_svs = SVSCollator(
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
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_svs,
        pin_memory=True,
    )
    

    with torch.no_grad():
        for (
            step,
            data_step,
        ) in enumerate(test_loader, 1):
            if args.db_joint:
                (
                    phone,
                    beat,
                    pitch,
                    spec,
                    real,
                    imag,
                    length,
                    chars,
                    char_len_list,
                    mel,
                    singer_id,
                    semitone,
                    filename_list
                ) = data_step

            else:
                print("No support for augmentation with args.db_joint == False")
                quit()

            singer_id = np.array(singer_id).reshape(
                np.shape(phone)[0], -1
            )  # [batch size, 1]
            singer_vec = singer_id.repeat(
                np.shape(phone)[1], axis=1
            )  # [batch size, length]
            singer_vec = torch.from_numpy(singer_vec).to(device)
            singer_id = torch.from_numpy(singer_id).to(device)

            phone = phone.to(device)
            beat = beat.to(device)
            pitch = pitch.to(device).float()
            if semitone is not None:
                semitone = semitone.to(device)
            spec = spec.to(device).float()
            mel = mel.to(device).float()
            real = real.to(device).float()
            imag = imag.to(device).float()
            length_mask = (length > 0).int().unsqueeze(2)
            length_mel_mask = length_mask.repeat(1, 1, mel.shape[2]).float()
            length_mask = length_mask.repeat(1, 1, spec.shape[2]).float()
            length_mask = length_mask.to(device)
            length_mel_mask = length_mel_mask.to(device)
            length = length.to(device)
            char_len_list = char_len_list.to(device)

            if not args.use_asr_post:
                chars = chars.to(device)
                char_len_list = char_len_list.to(device)
            else:
                phone = phone.float()

            if args.Hz2semitone:
                pitch = semitone

            if args.normalize:
                sepc_normalizer = GlobalMVN(args.stats_file)
                mel_normalizer = GlobalMVN(args.stats_mel_file)
                spec, _ = sepc_normalizer(spec, length)
                mel, _ = mel_normalizer(mel, length)

            len_list, _ = torch.max(length, dim=1)  # [len1, len2, len3, ...]
            len_list = len_list.cpu().detach().numpy()

            singer_out, phone_out, semitone_out = model(spec, len_list)

            # calculate num
            batch_size = np.shape(spec)[0]

            singer_id = singer_id.view(-1)  # [batch size]
            _, singer_predict = torch.max(singer_out, dim=1)    # [batch size]
            singer_correct = singer_predict.eq(singer_id).cpu().sum().numpy()

            for i in range(batch_size):
                phone_i = phone[i, : len_list[i], :].view(-1)  # [valid seq len]
                phone_out_i = phone_out[
                    i, : len_list[i], :
                ]  # [valid seq len, phone_size]
                _, phone_predict = torch.max(phone_out_i, dim=1)
                phone_correct = phone_predict.eq(phone_i).cpu().sum().numpy()

                semitone_i = semitone[i, : len_list[i], :].view(-1)  # [valid seq len]
                semitone_out_i = semitone_out[
                    i, : len_list[i], :
                ]  # [valid seq len, semitone_size]
                _, semitone_predict = torch.max(semitone_out_i, dim=1)
                semitone_correct = semitone_predict.eq(semitone_i).cpu().sum().numpy()

                with open(os.path.join(args.prediction_path, "filter_res.txt"), "a+") as f:
                    f.write(f"{filename_list[i]}|{singer_predict[i]}|{phone_correct}|{semitone_correct}|{len_list[i]}\n")

                end = time.time()
                
                logging.info(f"{filename_list[i]} -- sum_time: {(end - start_t_test)}s")



if __name__ == "__main__":

    args = parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f"{args}")

    data_filter(args)
    



