#!/usr/bin/env python3

# Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
#                                                   Haowen Qiu
#                                                   Fangjun Kuang)
#                2021  University of Chinese Academy of Sciences (author: Han Zhu)
# Apache 2.0

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import asdict

import k2
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lhotse.features.fbank import FbankConfig
from lhotse.utils import compute_num_frames
from lhotse.utils import fix_random_seed, nullcontext
from snowfall.common import describe, str2bool
from snowfall.common import load_checkpoint, save_checkpoint
from snowfall.common import save_training_info
from snowfall.common import setup_logger
from snowfall.data.librispeech import LibriSpeechAsrDataModule
from snowfall.dist import cleanup_dist
from snowfall.dist import setup_dist
from snowfall.lexicon import Lexicon
from snowfall.models import AcousticModel
from snowfall.models.conformer import Conformer
from snowfall.models.contextnet import ContextNet
from snowfall.models.tdnn_lstm import TdnnLstm1b  # alignment model
from snowfall.models.transformer import Noam, Transformer
from snowfall.objectives import LFMMILoss, encode_supervisions
from snowfall.training.diagnostics import measure_gradient_norms, optim_step_and_measure_param_change
from snowfall.training.mmi_graph import MmiTrainingGraphCompiler
from snowfall.training.mmi_graph import create_bigram_phone_lm

from lhotse.features.kaldi.layers import Wav2LogFilterBank
from denoiser.denoiser import DenoiserDefender

def feature_extraction(
        x: torch.Tensor,
        extractor: nn.Module,
        supervisions: dict = None,
        compute_gradient: bool = False,
) -> Tuple[torch.Tensor, dict]:
    """
    Do feature extraction and return feature, supervisions
    """

    # if compute_gradient:
    #     x.requires_grad = True

    feature = extractor(x)

    if supervisions is not None:
        start_frames = [
            compute_num_frames(sample.item() / 16000, extractor.frame_shift, 16000)
            for sample in supervisions['start_sample']
        ]
        supervisions['start_frame'] = torch.LongTensor(start_frames)

        num_frames = [
            compute_num_frames(sample.item() / 16000, extractor.frame_shift, 16000)
            for sample in supervisions['num_samples']
        ]
        supervisions['num_frames'] = torch.LongTensor(num_frames)

    return feature, supervisions


def forward_pass(feature,
                 supervisions,
                 supervision_segments,
                 texts,
                 P,
                 model,
                 ali_model,
                 device,
                 den_scale,
                 att_rate,
                 graph_compiler,
                 is_training,
                 global_batch_idx_train,
                 scaler, 
                 loss_fn,
                 ):
    feature = feature.permute(0, 2, 1)  # now feature is [N, C, T]
    grad_context = nullcontext if is_training else torch.no_grad
    with autocast(enabled=scaler.is_enabled()), grad_context():
        nnet_output, encoder_memory, memory_mask = model(feature, supervisions)
    # nnet_output is [N, C, T]
    nnet_output = nnet_output.permute(0, 2, 1)  # now nnet_output is [N, T, C]
    mmi_loss, tot_frames, all_frames = loss_fn(nnet_output, texts, supervision_segments)
    assert nnet_output.device == device, f'{nnet_output.device} != {device}'
    loss = (-mmi_loss) / (len(texts))
    return loss, tot_frames, all_frames


def fgsm_attack(audio,
                supervisions,
                supervision_segments,
                texts,
                P,
                model,
                ali_model,
                device,
                den_scale,
                att_rate,
                graph_compiler,
                is_training,
                global_batch_idx_train,
                scaler,
                loss_fn,
                feat_extractor,
                denoiser,
                eps=0.01):
    audio = audio.clone().to(device)
    eps = eps * audio.detach().abs().max().data
    if denoiser is not None:
        audio = denoiser(audio)
    feature, _ = feature_extraction(audio, feat_extractor, compute_gradient=True)
    loss, tot_frames, all_frames = forward_pass(feature, supervisions,
                                                supervision_segments,
                                                texts, P, model,
                                                ali_model,
                                                device, den_scale,
                                                att_rate, graph_compiler,
                                                is_training,
                                                global_batch_idx_train,
                                                scaler, loss_fn)
    scaler.scale(loss).backward()  # to get input gradients
    assert audio.grad is not None
    audio_adv = audio + audio.grad.data.sign() * eps
    audio_adv = torch.clamp(audio_adv, min=-1.0, max=1.0)
    audio = audio_adv.detach()
    return audio


def pgd_attack(audio,
               supervisions,
               supervision_segments,
               texts,
               P,
               model,
               ali_model,
               device,
               den_scale,
               att_rate,
               graph_compiler,
               is_training,
               global_batch_idx_train,
               scaler,
               loss_fn,
               feat_extractor,
               denoiser,
               eps=0.01,
               iters=7,
               rand_prob=0.8):
    from scipy.stats import loguniform
    audio = audio.clone().to(device)
    audio_ori = audio
    eps = loguniform.rvs(eps / 100, eps * 2, size=1)[0].item()
    eps = eps * audio.detach().abs().max().data
    iters = random.randint(1, iters)
    # 1.5 is a magic number to make it more likely for PGD to actually
    # reach the given epsilon for some samples
    alpha = 1.5 * (eps / iters) * audio.detach().abs().max().data
    if torch.rand(1) < rand_prob:
        rand_pert = torch.rand_like(audio) * 2 * eps - eps
        audio = audio + rand_pert
    audio = audio.detach()
    # print(audio.shape)
    for i in range(iters):
        audio.requires_grad = True
        if denoiser is not None:
            audio_denoised = denoiser(audio)
        else:
            audio_denoised = audio
        feature, _ = feature_extraction(audio_denoised, feat_extractor, compute_gradient=True)
        loss, tot_frames, all_frames = forward_pass(feature, supervisions,
                                                    supervision_segments,
                                                    texts, P, model,
                                                    ali_model,
                                                    device, den_scale,
                                                    att_rate, graph_compiler,
                                                    is_training,
                                                    global_batch_idx_train, scaler, loss_fn)
        scaler.scale(loss).backward()  # to get input gradients
        assert audio.grad is not None
        audio_adv = audio + audio.grad.data.sign() * alpha
        eta = torch.clamp(audio_adv - audio_ori, min=-eps, max=eps)
        audio = (audio_ori + eta)
        audio = torch.clamp(audio, min=-1.0, max=1.0)
        audio = audio.detach()

    return audio


def get_objf(batch: Dict,
             model: AcousticModel,
             ali_model: Optional[AcousticModel],
             P: k2.Fsa,
             device: torch.device,
             graph_compiler: MmiTrainingGraphCompiler,
             is_training: bool,
             is_update: bool,
             feat_extractor: nn.Module,
             denoiser: nn.Module,
             accum_grad: int = 1,
             den_scale: float = 1.0,
             att_rate: float = 0.0,
             tb_writer: Optional[SummaryWriter] = None,
             global_batch_idx_train: Optional[int] = None,
             optimizer: Optional[torch.optim.Optimizer] = None,
             scaler: Optional[GradScaler] = None,
             args=None):
    audio = batch['inputs'].clone().to(device)
    supervisions = batch['supervisions']
    _, supervisions = feature_extraction(audio, feat_extractor, supervisions, compute_gradient=False)
    supervision_segments = torch.stack(
        (supervisions['sequence_idx'],
         (((supervisions['start_frame'] - 1) // 2 - 1) // 2),
         (((supervisions['num_frames'] - 1) // 2 - 1) // 2)), 1).to(torch.int32)
    supervision_segments = torch.clamp(supervision_segments, min=0)
    indices = torch.argsort(supervision_segments[:, 2], descending=True)
    supervision_segments = supervision_segments[indices]

    texts = supervisions['text']
    texts = [texts[idx] for idx in indices]

    loss_fn = LFMMILoss(
        graph_compiler=graph_compiler,
        P=P,
        den_scale=den_scale
    )

    if is_training:
        # adversarial attack block
        if args.adv is not None:
            if args.adv == 'fgsm':
                audio = fgsm_attack(audio, supervisions,
                                    supervision_segments,
                                    texts, P, model,
                                    ali_model,
                                    device, den_scale,
                                    att_rate, graph_compiler,
                                    is_training,
                                    global_batch_idx_train,
                                    scaler, loss_fn,
                                    feat_extractor,
                                    denoiser,
                                    eps=args.fgsm_eps)
                optimizer.zero_grad()  # clean up gradients in model that were generated from adversary

            if args.adv == 'pgd':
                audio = pgd_attack(audio, supervisions,
                                   supervision_segments,
                                   texts, P, model,
                                   ali_model,
                                   device, den_scale,
                                   att_rate, graph_compiler,
                                   is_training,
                                   global_batch_idx_train,
                                   scaler, loss_fn,
                                   feat_extractor,
                                   denoiser,
                                   eps=args.pgd_eps,
                                   iters=args.pgd_iter,
                                   rand_prob=args.pgd_rand_prob)
                optimizer.zero_grad()  # clean up gradients in model that were generated from adversary

            if args.adv == 'm-pgd':  # a mixture of clean samples and adversarial samples
                audio_adv = pgd_attack(audio, supervisions,
                                       supervision_segments,
                                       texts, P, model,
                                       ali_model,
                                       device, den_scale,
                                       att_rate, graph_compiler,
                                       is_training,
                                       global_batch_idx_train,
                                       scaler, loss_fn,
                                       feat_extractor,
                                       denoiser,
                                       eps=args.pgd_eps,
                                       iters=args.pgd_iter,
                                       rand_prob=args.pgd_rand_prob)
                optimizer.zero_grad()  # clean up gradients in model that were generated from adversary
                if denoiser is not None:
                    audio_adv = denoiser(audio_adv)
                feature_adv, _ = feature_extraction(audio_adv, feat_extractor)
                loss, tot_frames, all_frames = forward_pass(feature_adv, supervisions,
                                                            supervision_segments,
                                                            texts, P, model,
                                                            ali_model,
                                                            device, den_scale,
                                                            att_rate, graph_compiler,
                                                            is_training,
                                                            global_batch_idx_train,
                                                            scaler, loss_fn
                                                            )
                scaler.scale(loss).backward()

    # forward and backward for paramters update
    if denoiser is not None:
        audio = denoiser(audio)
    feature, _ = feature_extraction(audio, feat_extractor, compute_gradient=False)
    loss, tot_frames, all_frames = forward_pass(feature, supervisions,
                                                supervision_segments,
                                                texts, P, model,
                                                ali_model,
                                                device, den_scale,
                                                att_rate, graph_compiler,
                                                is_training,
                                                global_batch_idx_train,
                                                scaler, loss_fn
                                                )
    if is_training:

        scaler.scale(loss).backward()

        if is_update:

            # def maybe_log_gradients(tag: str):
            #     if tb_writer is not None and global_batch_idx_train is not None and global_batch_idx_train % 200 == 0:
            #         tb_writer.add_scalars(
            #             tag,
            #             measure_gradient_norms(model, norm='l1'),
            #             global_step=global_batch_idx_train
            #         )

            # maybe_log_gradients('train/grad_norms')
            scaler.unscale_(optimizer)
            clip_grad_value_(model.parameters(), 5.0)
            # maybe_log_gradients('train/clipped_grad_norms')
            if tb_writer is not None and (global_batch_idx_train // accum_grad) % 200 == 0:
                # Once in a time we will perform a more costly diagnostic
                # to check the relative parameter change per minibatch.
                deltas = optim_step_and_measure_param_change(model, optimizer, scaler)
                tb_writer.add_scalars(
                    'train/relative_param_change_per_minibatch',
                    deltas,
                    global_step=global_batch_idx_train
                )
            else:
                scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()

    ans = loss.detach().cpu().item(), tot_frames.cpu().item(), all_frames.cpu().item()
    return ans


def get_validation_objf(dataloader: torch.utils.data.DataLoader,
                        model: AcousticModel,
                        ali_model: Optional[AcousticModel],
                        P: k2.Fsa,
                        device: torch.device,
                        graph_compiler: MmiTrainingGraphCompiler,
                        scaler: GradScaler,
                        feat_extractor: nn.Module,
                        denoiser: nn.Module,
                        den_scale: float = 1,
                        args=None,
                        ):
    total_objf = 0.
    total_frames = 0.  # for display only
    total_all_frames = 0.  # all frames including those seqs that failed.

    model.eval()

    from torchaudio.datasets.utils import bg_iterator
    for batch_idx, batch in enumerate(bg_iterator(dataloader, 2)):
        objf, frames, all_frames = get_objf(
            batch=batch,
            model=model,
            ali_model=ali_model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=False,
            is_update=False,
            feat_extractor=feat_extractor,
            denoiser=denoiser,
            den_scale=den_scale,
            scaler=scaler,
            args=args
        )
        total_objf += objf
        total_frames += frames
        total_all_frames += all_frames

    return total_objf, total_frames, total_all_frames


def train_one_epoch(dataloader: torch.utils.data.DataLoader,
                    valid_dataloader: torch.utils.data.DataLoader,
                    model: AcousticModel,
                    ali_model: Optional[AcousticModel],
                    denoiser: Optional[nn.Module],
                    P: k2.Fsa,
                    device: torch.device,
                    graph_compiler: MmiTrainingGraphCompiler,
                    optimizer: torch.optim.Optimizer,
                    accum_grad: int,
                    den_scale: float,
                    att_rate: float,
                    current_epoch: int,
                    tb_writer: SummaryWriter,
                    num_epochs: int,
                    global_batch_idx_train: int,
                    world_size: int,
                    scaler: GradScaler,
                    args=None,
                    ):
    """One epoch training and validation.

    Args:
        dataloader: Training dataloader
        valid_dataloader: Validation dataloader
        model: Acoustic model to be trained
        P: An FSA representing the bigram phone LM
        device: Training device, torch.device("cpu") or torch.device("cuda", device_id)
        graph_compiler: MMI training graph compiler
        optimizer: Training optimizer
        accum_grad: Number of gradient accumulation
        den_scale: Denominator scale in mmi loss
        att_rate: Attention loss rate, final loss is att_rate * att_loss + (1-att_rate) * other_loss
        current_epoch: current training epoch, for logging only
        tb_writer: tensorboard SummaryWriter
        num_epochs: total number of training epochs, for logging only
        global_batch_idx_train: global training batch index before this epoch, for logging only

    Returns:
        A tuple of 3 scalar:  (total_objf / total_frames, valid_average_objf, global_batch_idx_train)
        - `total_objf / total_frames` is the average training loss
        - `valid_average_objf` is the average validation loss
        - `global_batch_idx_train` is the global training batch index after this epoch
    """
    total_objf, total_frames, total_all_frames = 0., 0., 0.
    valid_average_objf = float('inf')
    time_waiting_for_batch = 0
    forward_count = 0
    prev_timestamp = datetime.now()
    
    fbank = Wav2LogFilterBank().to(device)

    model.train()
    if denoiser is not None:
        denoiser.model.train()
    for batch_idx, batch in enumerate(dataloader):
        forward_count += 1
        if forward_count == accum_grad:
            is_update = True
            forward_count = 0
        else:
            is_update = False

        global_batch_idx_train += 1
        timestamp = datetime.now()
        time_waiting_for_batch += (timestamp - prev_timestamp).total_seconds()

        if forward_count == 1 or accum_grad == 1:
            P.set_scores_stochastic_(model.module.P_scores)
            assert P.requires_grad is True

        curr_batch_objf, curr_batch_frames, curr_batch_all_frames = get_objf(
            batch=batch,
            model=model,
            ali_model=ali_model,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            is_training=True,
            is_update=is_update,
            feat_extractor=fbank,
            denoiser=denoiser,
            accum_grad=accum_grad,
            den_scale=den_scale,
            att_rate=att_rate,
            tb_writer=tb_writer,
            global_batch_idx_train=global_batch_idx_train,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
        )

        total_objf += curr_batch_objf
        total_frames += curr_batch_frames
        total_all_frames += curr_batch_all_frames

        if batch_idx % 10 == 0:
            logging.info(
                'batch {}, epoch {}/{} '
                'global average objf: {:.6f} over {} '
                'frames ({:.1f}% kept), current batch average objf: {:.6f} over {} frames ({:.1f}% kept) '
                'avg time waiting for batch {:.3f}s'.format(
                    batch_idx, current_epoch, num_epochs,
                    total_objf / total_frames, total_frames,
                    100.0 * total_frames / total_all_frames,
                    curr_batch_objf / (curr_batch_frames + 0.001),
                    curr_batch_frames,
                    100.0 * curr_batch_frames / curr_batch_all_frames,
                    time_waiting_for_batch / max(1, batch_idx)))

            if tb_writer is not None:
                tb_writer.add_scalar('train/global_average_objf',
                                     total_objf / total_frames, global_batch_idx_train)

                tb_writer.add_scalar('train/current_batch_average_objf',
                                     curr_batch_objf / (curr_batch_frames + 0.001),
                                     global_batch_idx_train)
            # if batch_idx >= 10:
            #    print("Exiting early to get profile info")
            #    sys.exit(0)

        if batch_idx > 0 and batch_idx % 200 == 0:
            total_valid_objf, total_valid_frames, total_valid_all_frames = get_validation_objf(
                dataloader=valid_dataloader,
                model=model,
                ali_model=ali_model,
                P=P,
                device=device,
                graph_compiler=graph_compiler,
                scaler=scaler,
                feat_extractor=fbank,
                denoiser=denoiser,
                args=args,
                )
            if world_size > 1:
                s = torch.tensor([
                    total_valid_objf, total_valid_frames,
                    total_valid_all_frames
                ]).to(device)

                dist.all_reduce(s, op=dist.ReduceOp.SUM)
                total_valid_objf, total_valid_frames, total_valid_all_frames = s.cpu().tolist()

            valid_average_objf = total_valid_objf / total_valid_frames
            model.train()
            logging.info(
                'Validation average objf: {:.6f} over {} frames ({:.1f}% kept)'
                    .format(valid_average_objf,
                            total_valid_frames,
                            100.0 * total_valid_frames / total_valid_all_frames))

            if tb_writer is not None:
                tb_writer.add_scalar('train/global_valid_average_objf',
                                     valid_average_objf,
                                     global_batch_idx_train)
                model.module.write_tensorboard_diagnostics(tb_writer, global_step=global_batch_idx_train)
        prev_timestamp = datetime.now()
    return total_objf / total_frames, valid_average_objf, global_batch_idx_train


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--world-size',
        type=int,
        default=1,
        help='Number of GPUs for DDP training.')
    parser.add_argument(
        '--master-port',
        type=int,
        default=12354,
        help='Master port to use for DDP training.')
    parser.add_argument(
        '--model-type',
        type=str,
        default="conformer",
        choices=["transformer", "conformer", "contextnet"],
        help="Model type.")
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=10,
        help="Number of training epochs.")
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help="Number of start epoch.")
    parser.add_argument(
        '--warm-step',
        type=int,
        default=5000,
        help='The number of warm-up steps for Noam optimizer.'
    )
    parser.add_argument(
        '--lr-factor',
        type=float,
        default=1.0,
        help='Learning rate factor for Noam optimizer.'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='weight decay (L2 penalty) for Noam optimizer.'
    )
    parser.add_argument(
        '--accum-grad',
        type=int,
        default=1,
        help="Number of gradient accumulation.")
    parser.add_argument(
        '--den-scale',
        type=float,
        default=1.0,
        help="denominator scale in mmi loss.")
    parser.add_argument(
        '--att-rate',
        type=float,
        default=0.0,
        help="Attention loss rate.")
    parser.add_argument(
        '--nhead',
        type=int,
        default=4,
        help="Number of attention heads in transformer.")
    parser.add_argument(
        '--attention-dim',
        type=int,
        default=256,
        help="Number of units in transformer attention layers.")
    parser.add_argument(
        '--tensorboard',
        type=str2bool,
        default=True,
        help='Should various information be logged in tensorboard.'
    )
    parser.add_argument(
        '--amp',
        type=str2bool,
        default=True,
        help='Should we use automatic mixed precision (AMP) training.'
    )
    parser.add_argument(
        '--use-ali-model',
        type=str2bool,
        default=False,
        help='If true, we assume that you have run ./ctc_train.py '
             'and you have some checkpoints inside the directory '
             'exp-lstm-adam-ctc-musan/ .'
             'It will use exp-lstm-adam-ctc-musan/epoch-{ali-model-epoch}.pt '
             'as the pre-trained alignment model'
    )
    parser.add_argument(
        '--ali-model-epoch',
        type=int,
        default=7,
        help='If --use-ali-model is True, load '
             'exp-lstm-adam-ctc-musan/epoch-{ali-model-epoch}.pt as the alignment model.'
             'Used only if --use-ali-model is True.'
    )
    parser.add_argument(
        '--adv',
        type=str,
        default=None,
        help='Adversarial training methods'
    )
    parser.add_argument(
        '--fgsm-eps',
        type=float,
        default=0.0,
        help='FGSM attack eps'
    )
    parser.add_argument(
        '--pgd-iter',
        type=int,
        default=7,
        help='Number of PGD iterations'
    )
    parser.add_argument(
        '--pgd-eps',
        type=float,
        default=0.0,
        help='PGD attack eps'
    )
    parser.add_argument(
        '--pgd-rand-prob',
        type=float,
        default=0.8,
        help='PGD attack random initialization prob'
    )
    parser.add_argument(
        '--fine-tune-mdl',
        type=str,
        default=None,
        help='Path to a pre-trained checkpoint. '
             'Overrides start_epoch etc. settings.'
    )
    parser.add_argument(
        '--discard-trainer-state',
        type=str2bool,
        default=False,
        help='Discards optimier, LR scheduler, grad scaler states '
             'if they are present in the checkpoint.'
    )
    parser.add_argument(
        '--denoiser-model-dir',
        type=str,
        default=None,
        help='denoiser model checkpoint dir')
    parser.add_argument(
        '--denoiser-model-ckpt',
        type=str,
        default=None,
        help='denoiser model checkpoint')
    parser.add_argument(
        '--exp',
        type=str,
        required=True,
        help="exp dir"
    )
    return parser


def run(rank, world_size, args):
    '''
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    '''
    model_type = args.model_type
    start_epoch = args.start_epoch
    num_epochs = args.num_epochs
    accum_grad = args.accum_grad
    den_scale = args.den_scale
    att_rate = args.att_rate

    fix_random_seed(42)
    setup_dist(rank, world_size, args.master_port)

    # exp_dir = Path('exp-' + model_type + '-noam-mmi-att-musan-sa-vgg-adv-' + str(args.adv) + '-4')
    exp_dir = Path(args.exp)
    setup_logger(f'{exp_dir}/log/log-train-{rank}')
    if args.tensorboard and rank == 0:
        tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard')
    else:
        tb_writer = None
    #  tb_writer = SummaryWriter(log_dir=f'{exp_dir}/tensorboard') if args.tensorboard and rank == 0 else None

    logging.info("Loading lexicon and symbol tables")
    lang_dir = Path('data/lang_nosp')
    lexicon = Lexicon(lang_dir)

    device_id = rank
    device = torch.device('cuda', device_id)

    graph_compiler = MmiTrainingGraphCompiler(
        lexicon=lexicon,
        device=device,
    )
    phone_ids = lexicon.phone_symbols()
    P = create_bigram_phone_lm(phone_ids)
    P.scores = torch.zeros_like(P.scores)
    P = P.to(device)

    librispeech = LibriSpeechAsrDataModule(args)
    train_dl = librispeech.train_dataloaders()
    valid_dl = librispeech.valid_dataloaders()

    if not torch.cuda.is_available():
        logging.error('No GPU detected!')
        sys.exit(-1)

    logging.info("About to create model")

    if att_rate != 0.0:
        num_decoder_layers = 6
    else:
        num_decoder_layers = 0

    if model_type == "transformer":
        model = Transformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=True)
    elif model_type == "conformer":
        model = Conformer(
            num_features=80,
            nhead=args.nhead,
            d_model=args.attention_dim,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4,
            num_decoder_layers=num_decoder_layers,
            vgg_frontend=True)
    elif model_type == "contextnet":
        model = ContextNet(
            num_features=80,
            num_classes=len(phone_ids) + 1)  # +1 for the blank symbol
    else:
        raise NotImplementedError("Model of type " + str(model_type) + " is not implemented")

    model.P_scores = nn.Parameter(P.scores.clone(), requires_grad=True)

    model.to(device)
    describe(model)

    model = DDP(model, device_ids=[rank])

    # Now for the aligment model, if any
    if args.use_ali_model:
        ali_model = TdnnLstm1b(
            num_features=80,
            num_classes=len(phone_ids) + 1,  # +1 for the blank symbol
            subsampling_factor=4)

        ali_model_fname = Path(f'exp-lstm-adam-ctc-musan/epoch-{args.ali_model_epoch}.pt')
        assert ali_model_fname.is_file(), \
            f'ali model filename {ali_model_fname} does not exist!'
        ali_model.load_state_dict(torch.load(ali_model_fname, map_location='cpu')['state_dict'])
        ali_model.to(device)

        ali_model.eval()
        ali_model.requires_grad_(False)
        logging.info(f'Use ali_model: {ali_model_fname}')
    else:
        ali_model = None
        logging.info('No ali_model')
        
    if args.denoiser_model_ckpt:
        denoiser = DenoiserDefender(args.denoiser_model_dir, args.denoiser_model_ckpt, device)
    # denoiser = None

        
    optimizer = Noam(model.parameters(),
                     model_size=args.attention_dim,
                     factor=args.lr_factor,
                     warm_step=args.warm_step,
                     weight_decay=args.weight_decay)

    scaler = GradScaler(enabled=args.amp)

    best_objf = np.inf
    best_valid_objf = np.inf
    best_epoch = start_epoch
    best_model_path = os.path.join(exp_dir, 'best_asr_model.pt')
    if denoiser is not None:
        best_denoiser_path = os.path.join(exp_dir, 'best_denoiser_model.pt')
    best_epoch_info_filename = os.path.join(exp_dir, 'best-epoch-info')
    global_batch_idx_train = 0  # for logging only

    if start_epoch > 0 or args.fine_tune_mdl:
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(start_epoch - 1))
        if args.fine_tune_mdl:
            model_path = args.fine_tune_mdl
            logging.info(f'Reading pre-trained model from {model_path} for fine-tuning. '
                         f'You can use --start-epoch to control the learning rate schedule...')
        if args.discard_trainer_state:
            ckpt = load_checkpoint(filename=model_path, model=model)
        else:
            ckpt = load_checkpoint(filename=model_path, model=model, optimizer=optimizer, scaler=scaler)
        if all(x in ckpt for x in 'objf valid_objf global_batch_idx_train epoch best_objf best_balid_objf'.split()):
            best_objf = ckpt['objf']
            best_valid_objf = ckpt['valid_objf']
            global_batch_idx_train = ckpt['global_batch_idx_train']
            logging.info(f"epoch = {ckpt['epoch']}, objf = {best_objf}, valid_objf = {best_valid_objf}")

    for epoch in range(start_epoch, num_epochs):
        train_dl.sampler.set_epoch(epoch)
        curr_learning_rate = optimizer._rate
        if tb_writer is not None:
            tb_writer.add_scalar('train/learning_rate', curr_learning_rate, global_batch_idx_train)
            tb_writer.add_scalar('train/epoch', epoch, global_batch_idx_train)

        logging.info('epoch {}, learning rate {}'.format(epoch, curr_learning_rate))
        objf, valid_objf, global_batch_idx_train = train_one_epoch(
            dataloader=train_dl,
            valid_dataloader=valid_dl,
            model=model,
            ali_model=ali_model,
            denoiser=denoiser,
            P=P,
            device=device,
            graph_compiler=graph_compiler,
            optimizer=optimizer,
            accum_grad=accum_grad,
            den_scale=den_scale,
            att_rate=att_rate,
            current_epoch=epoch,
            tb_writer=tb_writer,
            num_epochs=num_epochs,
            global_batch_idx_train=global_batch_idx_train,
            world_size=world_size,
            scaler=scaler,
            args=args,
        )
        # the lower, the better
        if valid_objf < best_valid_objf:
            best_valid_objf = valid_objf
            best_objf = objf
            best_epoch = epoch
            save_checkpoint(filename=best_model_path,
                            optimizer=None,
                            scheduler=None,
                            scaler=None,
                            model=model,
                            epoch=epoch,
                            learning_rate=curr_learning_rate,
                            objf=objf,
                            valid_objf=valid_objf,
                            global_batch_idx_train=global_batch_idx_train,
                            local_rank=rank)
            if denoiser is not None:
                torch.save(denoiser.model.state_dict(), best_denoiser_path)
            save_training_info(filename=best_epoch_info_filename,
                               model_path=best_model_path,
                               current_epoch=epoch,
                               learning_rate=curr_learning_rate,
                               objf=objf,
                               best_objf=best_objf,
                               valid_objf=valid_objf,
                               best_valid_objf=best_valid_objf,
                               best_epoch=best_epoch,
                               local_rank=rank)

        # we always save the model for every epoch
        model_path = os.path.join(exp_dir, 'epoch-{}.pt'.format(epoch))
        if denoiser is not None:
            denoiser_path = os.path.join(exp_dir, 'epoch-{}-denoiser.pt'.format(epoch))
        save_checkpoint(filename=model_path,
                        optimizer=optimizer,
                        scheduler=None,
                        scaler=scaler,
                        model=model,
                        epoch=epoch,
                        learning_rate=curr_learning_rate,
                        objf=objf,
                        valid_objf=valid_objf,
                        global_batch_idx_train=global_batch_idx_train,
                        local_rank=rank)
        if denoiser is not None:
            torch.save(denoiser.model.state_dict(), denoiser_path)
        epoch_info_filename = os.path.join(exp_dir, 'epoch-{}-info'.format(epoch))
        save_training_info(filename=epoch_info_filename,
                           model_path=model_path,
                           current_epoch=epoch,
                           learning_rate=curr_learning_rate,
                           objf=objf,
                           best_objf=best_objf,
                           valid_objf=valid_objf,
                           best_valid_objf=best_valid_objf,
                           best_epoch=best_epoch,
                           local_rank=rank)

    logging.warning('Done')
    torch.distributed.barrier()
    cleanup_dist()


def main():
    parser = get_parser()
    LibriSpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    world_size = args.world_size
    assert world_size >= 1
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)


torch.set_num_threads(1)
torch.set_num_interop_threads(1)

logging.info = print

if __name__ == '__main__':
    main()
