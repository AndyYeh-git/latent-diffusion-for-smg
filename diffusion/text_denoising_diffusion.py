import math
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
import os
import numpy as np
import json
import argparse
from datetime import timedelta
import time

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from tqdm.auto import tqdm
from ema_pytorch import EMA

from transformers import get_scheduler, BartConfig, T5Config, T5ForConditionalGeneration, BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
import wandb

import diffusion.constant as constant
import diffusion.optimizer as optimizer
from dataset_utils.create_dataloader import get_dataset, MelodyDataset, get_dataloader

from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from latent_models.latent_utils import get_latent_model
from evaluation import evaluation

from itertools import cycle, islice
from music21 import converter, midi, stream, meter

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalize variance of noised latent, if scale is not 1

def normalize_z_t_variance(z_t, mask, eps = 1e-5):
    std = rearrange([reduce(z_t[i][:torch.sum(mask[i])], 'l d -> 1 1', partial(torch.std, unbiased = False)) for i in range(z_t.shape[0])], 'b 1 1 -> b 1 1')
    return z_t / std.clamp(min = eps)    

# noise schedules

def simple_linear_schedule(t, clip_min = 1e-9):
    return (1 - t).clamp(min = clip_min)

def beta_linear_schedule(t, clip_min = 1e-9):
    return torch.exp(-1e-4 - 10 * (t ** 2)).clamp(min = clip_min, max = 1.)

def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

def sigmoid_schedule(t, start = -3, end = 3, tau = 1, clamp_min = 1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min = clamp_min, max = 1.)

def tand_schedule(t, d = 9, clamp_min = 1e-9):
    gamma = 1 / (1 + (torch.tan(math.pi * t / 2) ** 2) * (d ** 2))
    return gamma.clamp_(min = clamp_min, max = 1.)

def laplace_noise_schedule(t, mu=0.0, b=0.5, clamp_min = 1e-9):

    lmb = mu - b * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))
    snr = torch.exp(lmb).clamp_(max = 1e18)
    alpha = snr / (1 + snr)

    return alpha.clamp_(min = clamp_min, max = 1.)

# converting gamma to alpha, sigma or logsnr

def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

def alpha_to_shifted_log_snr(alpha, scale = 1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        max_seq_len,
        sampling_timesteps = 250,
        loss_type = 'l1',
        objective = 'pred_noise',
        train_schedule = 'cosine',
        sampling_schedule = None,
        scale = 1.,
        sampler = 'ddpm',
        method = 'fixed',
        train_prob_self_cond = 0.5,
        seq2seq_unconditional_prob = 0.1,
    ):
        super().__init__()
        assert sampler in {'ddim', 'ddpm', 'dpmpp'}, 'sampler must be one of ddim, ddpm, dpmpp'
        self.sampler = sampler
        self.method = method

        self.diffusion_model = model
        if self.diffusion_model.class_conditional:
            if self.diffusion_model.class_unconditional_prob > 0:
                self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.class_unconditional_prob)

        self.latent_dim = self.diffusion_model.latent_dim
        self.seq2seq_context_dim = self.diffusion_model.seq2seq_context_dim
        self.self_condition = self.diffusion_model.self_condition

        self.max_seq_len = max_seq_len
        self.l2_normalize = False

        self.objective = objective

        self.loss_type = loss_type

        assert objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_v_dual'}, 'objective must be one of pred_noise, pred_x0, pred_v, pred_v_dual'

        if train_schedule == "simple_linear":
            alpha_schedule = simple_linear_schedule
        elif train_schedule == "beta_linear":
            alpha_schedule = beta_linear_schedule
        elif train_schedule == "cosine":
            alpha_schedule = cosine_schedule
        elif train_schedule == "sigmoid":
            alpha_schedule = sigmoid_schedule
        elif train_schedule == "tand":
            alpha_schedule = tand_schedule
        elif train_schedule == "laplace":
            alpha_schedule = laplace_noise_schedule
        else:
            raise ValueError(f'invalid noise schedule {train_schedule}')
        
        if train_schedule == "tand" or train_schedule ==  "laplace":
            self.train_schedule = alpha_schedule
        else:
            self.train_schedule = partial(time_to_alpha, alpha_schedule=alpha_schedule, scale=scale)

        # Sampling schedule
        if sampling_schedule is None:
            sampling_alpha_schedule = None
        elif sampling_schedule == "simple_linear":
            sampling_alpha_schedule = simple_linear_schedule
        elif sampling_schedule == "beta_linear":
            sampling_alpha_schedule = beta_linear_schedule
        elif sampling_schedule == "cosine":
            sampling_alpha_schedule = cosine_schedule
        elif sampling_schedule == "sigmoid":
            sampling_alpha_schedule = sigmoid_schedule
        elif sampling_schedule == "tand":
            sampling_alpha_schedule = tand_schedule
        elif sampling_schedule == "laplace":
            sampling_alpha_schedule = laplace_noise_schedule
        else:
            raise ValueError(f'invalid sampling schedule {sampling_schedule}')
        
        if exists(sampling_alpha_schedule):
            if sampling_schedule == "tand" or sampling_schedule == "laplace":
                self.sampling_schedule = sampling_alpha_schedule
            else:
                self.sampling_schedule = partial(time_to_alpha, alpha_schedule=sampling_alpha_schedule, scale=scale)
        else:
            self.sampling_schedule = self.train_schedule

        # the main finding presented in Ting Chen's paper - that higher resolution images requires more noise for better training
        
        self.scale = scale

        # gamma schedules

        self.sampling_timesteps = sampling_timesteps

        # probability for self conditioning during training

        self.train_prob_self_cond = train_prob_self_cond
        self.seq2seq_unconditional_prob = seq2seq_unconditional_prob

        # Buffers for latent mean and scale values
        self.register_buffer('latent_mean', torch.tensor([0]*self.seq2seq_context_dim).to(torch.float32))
        self.register_buffer('latent_scale', torch.tensor(1).to(torch.float32))
        self.register_buffer('seq2seq_latent_mean', torch.tensor([0]*self.seq2seq_context_dim).to(torch.float32))
        self.register_buffer('seq2seq_latent_scale', torch.tensor(1).to(torch.float32))

    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1-alpha).sqrt() * noise) / alpha.sqrt().clamp(min = 1e-8)
        
    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1-alpha).sqrt().clamp(min = 1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1-alpha).sqrt() * v

        return x
    
    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1-alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps
    
    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x* (1-alpha).sqrt()

        return v

    def normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.latent_mean)/(self.latent_scale).clamp(min=eps)
    
    def unnormalize_latent(self, x_start):
        eps = 1e-5 
        
        return x_start*(self.latent_scale.clamp(min=eps))+self.latent_mean
    
    def seq2seq_normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.seq2seq_latent_mean)/(self.seq2seq_latent_scale).clamp(min=eps)
    
    def seq2seq_unnormalize_latent(self, x_start):
        eps = 1e-5 
        
        return x_start*(self.seq2seq_latent_scale.clamp(min=eps))+self.seq2seq_latent_mean

    def diffusion_model_predictions(self, z_t, mask, t, *, x_self_cond = None,  class_id=None, seq2seq_cond=None,
                                    seq2seq_mask=None, sampling=False, cls_free_guidance=1.0, l2_normalize=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        if cls_free_guidance!=1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.num_classes)
            else:
                unc_class_id = None
            unc_model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=unc_class_id, seq2seq_cond=None, seq2seq_mask=None)
            model_output = model_output*cls_free_guidance + unc_model_output*(1-cls_free_guidance)

        pred_v = None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(z_t, t, pred_noise, sampling=sampling)
        elif self.objective =='pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)
        elif self.objective == 'pred_v':
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f'invalid objective {self.objective}')
        if l2_normalize:
            assert sampling
            x_start = F.normalize(x_start, dim=-1) * math.sqrt(x_start.shape[-1])
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)

        return ModelPrediction(pred_noise, x_start, pred_v)

    def get_sampling_timesteps(self, batch, *, device, invert = False):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        if invert:
            times = times.flip(dims = (0,))
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddim_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0,
                    l2_normalize=False, invert=False, z_t=None, infill=False, outfill=False):
        #print('DDIM sampling')
        batch, device = shape[0], next(self.diffusion_model.parameters()).device
        
        time_pairs = self.get_sampling_timesteps(batch, device = device, invert=invert)
        
        if invert:
            assert exists(z_t)

        if infill:
            assert exists(z_t) and not outfill
            fix_zt = z_t.clone()
            if self.method == "fixed":
                z_t =  torch.cat((z_t[:,:8].clone(), torch.randn((shape[0], shape[1]//2, shape[2]), device=device), z_t[:,-8:].clone()), dim=1)
            else:
                z_t = torch.randn(shape, device=device)
        elif outfill:
            assert exists(z_t) and not infill
            fix_zt = z_t.clone()
            if self.method == "fixed":
                z_t =  torch.cat((z_t[:,:16].clone(), torch.randn((shape[0], shape[1]//2, shape[2]), device=device)), dim=1)
            else:
                z_t = torch.randn(shape, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model or lengths == None:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps, disable=True):
            # get predicted x0

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask,
                                                            sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
         
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise
            
            if (not invert) and time_next[0] <= 0:
                if infill:
                    z_t =  torch.cat((fix_zt[:,:8].clone(), x_start[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                elif outfill:
                    z_t =  torch.cat((fix_zt[:,:16].clone(), x_start[:,16:].clone()), dim=1)
                else:
                    z_t = x_start
                continue
            if invert and time_next[0] >= 1:
                z_t = eps
                continue
            
            # get noise
            
            z_t = x_start * alpha_next.sqrt() + eps * (1-alpha_next).sqrt()

            if infill:
                if self.method == "fixed":
                    z_t =  torch.cat((fix_zt[:,:8].clone(), z_t[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                else:
                    noise = torch.randn_like(z_t)
                    noisy_fix_zt = fix_zt * alpha.sqrt() + noise * (1-alpha).sqrt()
                    if self.method == "hybrid":
                        if time_next[0] > 0.5:
                            z_t =  torch.cat((noisy_fix_zt[:,:8].clone(), z_t[:,8:24].clone(), noisy_fix_zt[:,-8:].clone()), dim=1)
                        else:
                            z_t =  torch.cat((fix_zt[:,:8].clone(), z_t[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                    else:
                        z_t =  torch.cat((noisy_fix_zt[:,:8].clone(), z_t[:,8:24].clone(), noisy_fix_zt[:,-8:].clone()), dim=1)
                
            elif outfill:
                if self.method == "fixed":
                    z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                else:
                    noise = torch.randn_like(z_t)
                    noisy_fix_zt = fix_zt * alpha.sqrt() + noise * (1-alpha).sqrt()
                    if self.method == "hybrid":
                        if time_next[0] > 0.5:
                            z_t =  torch.cat((noisy_fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                        else:
                            z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                    else:
                        z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
            
        return (z_t, mask)

    @torch.no_grad()
    def ddpm_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0,
                    l2_normalize=False, invert=False, z_t=None, infill=False, outfill=False):
        batch, device = shape[0], next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        if infill:
            assert exists(z_t) and not outfill
            fix_zt = z_t.clone()
            if self.method == "fixed":
                z_t =  torch.cat((z_t[:,:8].clone(), torch.randn((shape[0], shape[1]//2, shape[2]), device=device), z_t[:,-8:].clone()), dim=1)
            else:
                z_t = torch.randn(shape, device=device)
        elif outfill:
            assert exists(z_t) and not infill
            fix_zt = z_t.clone()
            if self.method == "fixed":
                z_t =  torch.cat((z_t[:,:16].clone(), torch.randn((shape[0], shape[1]//2, shape[2]), device=device)), dim=1)
            else:
                z_t = torch.randn(shape, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model or lengths == None:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps, disable=True):
            # get predicted x0

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask,
                                                            sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
            
            # get alpha sigma of time and next time

            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            alpha_now = alpha/alpha_next
            
            # calculate x0 and noise

            x_start = model_output.pred_x_start

            eps = model_output.pred_noise
            
            if time_next[0] <= 0:
                if infill:
                    z_t =  torch.cat((fix_zt[:,:8].clone(), x_start[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                elif outfill:
                    z_t =  torch.cat((fix_zt[:,:16].clone(), x_start[:,16:].clone()), dim=1)
                else:
                    z_t = x_start
                continue         
            
            # get noise

            noise = torch.randn_like(z_t)
            
            z_t = 1/alpha_now.sqrt() * (z_t - (1-alpha_now)/(1-alpha).sqrt() * eps) + torch.sqrt(1 - alpha_now) * noise

            if infill:
                if self.method == "fixed":
                    z_t =  torch.cat((fix_zt[:,:8].clone(), z_t[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                else:
                    noise = torch.randn_like(z_t)
                    noisy_fix_zt = fix_zt * alpha.sqrt() + noise * (1-alpha).sqrt()
                    if self.method == "hybrid":
                        if time_next[0] > 0.5:
                            z_t =  torch.cat((noisy_fix_zt[:,:8].clone(), z_t[:,8:24].clone(), noisy_fix_zt[:,-8:].clone()), dim=1)
                        else:
                            z_t =  torch.cat((fix_zt[:,:8].clone(), z_t[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                    else:
                        z_t =  torch.cat((noisy_fix_zt[:,:8].clone(), z_t[:,8:24].clone(), noisy_fix_zt[:,-8:].clone()), dim=1)
                
            elif outfill:
                if self.method == "fixed":
                    z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                else:
                    noise = torch.randn_like(z_t)
                    noisy_fix_zt = fix_zt * alpha.sqrt() + noise * (1-alpha).sqrt()
                    if self.method == "hybrid":
                        if time_next[0] > 0.5:
                            z_t =  torch.cat((noisy_fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                        else:
                            z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                    else:
                        z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)

        return (z_t, mask)

    @torch.no_grad()
    def dpmpp_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0,
                     l2_normalize=False, invert=False, z_t=None, infill=False, outfill=False):
        batch, device = shape[0], next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        if infill:
            assert exists(z_t) and not outfill
            fix_zt = z_t.clone()
            if self.method == "fixed":
                z_t =  torch.cat((z_t[:,:8].clone(), torch.randn((shape[0], shape[1]//2, shape[2]), device=device), z_t[:,-8:].clone()), dim=1)
            else:
                z_t = torch.randn(shape, device=device)
        elif outfill:
            assert exists(z_t) and not infill
            fix_zt = z_t.clone()
            if self.method == "fixed":
                z_t =  torch.cat((z_t[:,:16].clone(), torch.randn((shape[0], shape[1]//2, shape[2]), device=device)), dim=1)
            else:
                z_t = torch.randn(shape, device=device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model or lengths == None:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        old_pred_x = []
        old_hs = []

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps, disable=True):          
            # get predicted x0

            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask,
                                                            sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)

            # get alpha sigma of time and next time
            
            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))
            sigma, sigma_next = 1-alpha, 1-alpha_next

            alpha_now = alpha/alpha_next
            
            lambda_now = ((log(alpha) - log(1-alpha))/2)
            lambda_next = ((log(alpha_next) - log(1-alpha_next))/2)
            h = lambda_next - lambda_now

            # calculate x0 and noise
            if time_next[0] <= 0:
                if infill:
                    z_t =  torch.cat((fix_zt[:,:8].clone(), x_start[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                elif outfill:
                    z_t =  torch.cat((fix_zt[:,:16].clone(), x_start[:,16:].clone()), dim=1)
                else:
                    z_t = x_start
                continue  

            x_start = model_output.pred_x_start

            phi_1 = torch.expm1(-h)
            if len(old_pred_x) < 2:
                denoised_x = x_start
            else:
                h = lambda_next - lambda_now
                h_0 = old_hs[-1]
                r0 = h_0/h
                gamma = -1/(2*r0)
                denoised_x = (1-gamma)*x_start + gamma*old_pred_x[-1]

            z_t = (sigma_next.sqrt()/sigma.sqrt()) * z_t - alpha_next.sqrt() * phi_1 * denoised_x

            if infill:
                if self.method == "fixed":
                    z_t =  torch.cat((fix_zt[:,:8].clone(), z_t[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                else:
                    noise = torch.randn_like(z_t)
                    noisy_fix_zt = fix_zt * alpha.sqrt() + noise * (1-alpha).sqrt()
                    if self.method == "hybrid":
                        if time_next[0] > 0.5:
                            z_t =  torch.cat((noisy_fix_zt[:,:8].clone(), z_t[:,8:24].clone(), noisy_fix_zt[:,-8:].clone()), dim=1)
                        else:
                            z_t =  torch.cat((fix_zt[:,:8].clone(), z_t[:,8:24].clone(), fix_zt[:,-8:].clone()), dim=1)
                    else:
                        z_t =  torch.cat((noisy_fix_zt[:,:8].clone(), z_t[:,8:24].clone(), noisy_fix_zt[:,-8:].clone()), dim=1)
                
            elif outfill:
                if self.method == "fixed":
                    z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                else:
                    noise = torch.randn_like(z_t)
                    noisy_fix_zt = fix_zt * alpha.sqrt() + noise * (1-alpha).sqrt()
                    if self.method == "hybrid":
                        if time_next[0] > 0.5:
                            z_t =  torch.cat((noisy_fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                        else:
                            z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)
                    else:
                        z_t =  torch.cat((fix_zt[:,:16].clone(), z_t[:,16:].clone()), dim=1)

        return (z_t, mask)    

    @torch.no_grad()
    def sample(self, batch_size, length, class_id=None, seq2seq_cond=None, seq2seq_mask=None, cls_free_guidance=1.0,
               l2_normalize=False, z_t=None, infill=False, outfill=False):
        max_seq_len, latent_dim = self.max_seq_len, self.latent_dim
        
        if self.sampler == 'ddim':
            sample_fn = self.ddim_sample
        elif self.sampler == 'ddpm':
            sample_fn = self.ddpm_sample
        elif self.sampler == 'dpmpp':
            sample_fn = self.dpmpp_sample
        else:
            raise ValueError(f'invalid sampler {self.sampler}')
        return sample_fn((batch_size, max_seq_len, latent_dim), length, class_id, seq2seq_cond, seq2seq_mask,
                         cls_free_guidance, l2_normalize, z_t=z_t, infill=infill, outfill=outfill)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, *args, **kwargs):
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.max_seq_len
        assert l == max_seq_len, f'length must be {self.max_seq_len}'
        
        # sample random times

        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)
        # noise sample

        noise = torch.randn_like(txt_latent)

        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(txt_latent, alpha)

        z_t = alpha.sqrt() * txt_latent + (1-alpha).sqrt() * noise

        # Perform unconditional generation with some probability
        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            assert exists(class_id)
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes

        self_cond = None

        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(z_t, mask, times, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                self_cond = model_output.pred_x_start.detach()
                if self.l2_normalize:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])              

        # predict and take gradient step

        predictions = self.diffusion_model_predictions(z_t, mask, times, x_self_cond=self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)          
        if self.objective == 'pred_x0':
            target = txt_latent
            pred = predictions.pred_x_start
        elif self.objective == 'pred_noise':
            target = noise
            pred = predictions.pred_noise
        elif self.objective == 'pred_v':
            target = alpha.sqrt() * noise - (1-alpha).sqrt() * txt_latent
            assert exists(predictions.pred_v)
            pred = predictions.pred_v
            
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')

        if return_x_start:
            return loss.mean(), predictions.pred_x_start
        return loss.mean()

# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        diffusion,
        *,
        train_batch_size = 4,
        eval_batch_size = 4,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 1000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.999),
        adam_weight_decay = 0.01,
        save_and_sample_every = 5000,
        num_samples = 25,
        seq2seq_candidates = 10,
        seq2seq_train_context_encoder = False,
        results_folder = './results',
        amp = False,
        mixed_precision = 'no',
        decoding_loss = False,
        decoding_loss_weight = 1.0,
    ):
        super().__init__()

        set_seeds(args.seed)

        self.args = args

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))

        self.accelerator = Accelerator(
            mixed_precision = mixed_precision,
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs, init_process_kwargs]
        )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = file_utils.get_output_dir(args)
                with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                    json.dump(args.__dict__, f, indent=2)
            results_folder = args.output_dir
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})        

        self.diffusion = diffusion
        self.decoding_loss = decoding_loss
        self.decoding_loss_weight = decoding_loss_weight

        self.num_samples = num_samples
        self.seq2seq_candidates = seq2seq_candidates
        self.save_and_sample_every = save_and_sample_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.max_seq_len = diffusion.max_seq_len

        self.latent_model_path = args.latent_model_path

        self.enc_dec_model = args.enc_dec_model

        if 'bart' in args.enc_dec_model:
            # Init Bart model Config
            config = BartConfig.from_pretrained(args.enc_dec_model)
            config.max_position_embeddings = max(self.args.max_seq_len, 1024)
            config.vocab_size=args.vocab_size
            # Initialize the Bart model
            self.bart_model = BartForConditionalGeneration(config = config)
            model_type = 'bart'
        elif 'bert' in args.enc_dec_model:
            # Init Encoder decoder Config
            config = BertGenerationConfig.from_pretrained(args.enc_dec_model)
            config.max_position_embeddings = max(self.args.max_seq_len, config.max_position_embeddings)
            config.vocab_size=args.vocab_size
            config.bos_token_id=101
            config.eos_token_id=102
            # Init Encoder-decoder model
            encoder = BertGenerationEncoder(config = config)
            # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
            config.is_decoder=True
            decoder = BertGenerationDecoder(config = config)
            # Initialize the EncoderDecoderModel
            self.model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
            model_type = 'bert'
        elif 'flan-t5' in args.enc_dec_model:
            config = T5Config.from_pretrained(args.enc_dec_model)
            config.vocab_size=args.vocab_size
            self.bart_model = T5ForConditionalGeneration(config = config, torch_dtype=torch.bfloat16)
            model_type = 't5'
        else:
            raise ValueError(f'invalid enc_dec_model {args.enc_dec_model}')

        self.diffusion.using_latent_model = False
        self.seq2seq = self.diffusion.diffusion_model.seq2seq
        self.class_conditional = self.diffusion.diffusion_model.class_conditional
        self.seq2seq_unconditional_prob = self.diffusion.seq2seq_unconditional_prob
        self.best_seq2seq_metric = 0
        if self.seq2seq:
            self.diffusion.context_encoder = self.bart_model.get_encoder()
            self.seq2seq_train_context_encoder = seq2seq_train_context_encoder
            if seq2seq_train_context_encoder:
                for param in self.diffusion.context_encoder.parameters():
                    param.requires_grad = True
            else:
                for param in self.diffusion.context_encoder.parameters():
                    param.requires_grad = False
        if args.latent_model_path:
            device = self.accelerator.device
            with open(os.path.join(args.latent_model_path, 'args.json'), 'rt') as f:
                latent_model_args = json.load(f)
            latent_argparse = argparse.Namespace(**latent_model_args)
            self.bart_model, _ = get_latent_model(latent_argparse)
            data = torch.load(os.path.join(args.latent_model_path, 'model.pt'), map_location=device, weights_only=True)
            self.bart_model.load_state_dict(data['model'])
            self.diffusion.max_seq_len = self.bart_model.num_encoder_latents * args.seg
            self.num_encoder_latents = self.bart_model.num_encoder_latents
            self.diffusion.using_latent_model = True       
            self.diffusion.l2_normalize = (hasattr(self.bart_model, 'l2_normalize_latents') and self.bart_model.l2_normalize_latents)
            if self.diffusion.l2_normalize:
                assert not args.normalize_latent
        for param in self.bart_model.parameters():
            param.requires_grad = False
        
        self.using_latent_model = self.diffusion.using_latent_model
        self.bart_model.eval()            

        # Create the dataset        
        self.train_data, self.val_data = get_dataset(args)

        self.train_dataset = MelodyDataset(self.args, self.train_data)    
        self.train_dataloader = get_dataloader(args, self.train_dataset, self.bart_model.config, shuffle=True)

        self.val_dataset = MelodyDataset(self.args, self.val_data)
        self.val_dataloader = get_dataloader(args, self.val_dataset, self.bart_model.config, shuffle=True)

        self.test_dataset = MelodyDataset(self.args, self.val_data)
        self.test_dataloader = get_dataloader(args, self.test_dataset, self.bart_model.config, shuffle=False)

        self.train_val_dataset = MelodyDataset(self.args, {key: value[:1000] for key, value in self.train_data.items()})    
        self.train_val_dataloader = get_dataloader(args, self.train_val_dataset, self.bart_model.config, shuffle=False)

        if args.normalize_latent:
            if args.L == None:
                if args.M == None:
                    dataset_name = args.dataset_name
                else:
                    dataset_name = f"{args.dataset_name}/M{args.M.replace('/', '_')}"
            else:
                dataset_name = f"{args.dataset_name}/L{args.L.replace('/', '_')}_M{args.M.replace('/', '_')}"
            path = os.path.join("dataset_mean", dataset_name)

            if args.min != None:
                txt = os.path.join(path, f"dataset_mean_{args.min}_{model_type}.txt")
            else:
                txt = os.path.join(path, f"dataset_mean_{model_type}.txt")
        
            with open(txt, 'r') as f:
                mean_std = f.read()
                diffusion_latent_mean = mean_std.split("[")[1].split("]\n")[0]
                diffusion_latent_scale = mean_std.split("[")[2].split("]\n")[0]
                if self.seq2seq:
                    seq2seq_latent_mean = mean_std.split("[")[3].split("]\n")[0]
                    seq2seq_latent_scale = mean_std.split("[")[4].split("]\n")[0]
            
            self.diffusion.latent_mean = torch.tensor(np.fromstring(diffusion_latent_mean, dtype=float, sep=' ')).float()
            self.diffusion.latent_scale = torch.tensor(np.fromstring(diffusion_latent_scale, dtype=float, sep=' ')).float()
        
            if self.seq2seq:
                self.diffusion.seq2seq_latent_mean = torch.tensor(np.fromstring(seq2seq_latent_mean, dtype=float, sep=' ')).float()
                self.diffusion.seq2seq_latent_scale = torch.tensor(np.fromstring(seq2seq_latent_scale, dtype=float, sep=' ')).float()
        
        if args.eval_test:
            self.num_samples = min(self.num_samples, len(self.val_data['text']))
            print(f'Using {self.num_samples} samples for evaluation')
        else:
            self.num_samples = min(self.num_samples, len(self.val_data['text']))
            print(f'Using {self.num_samples} samples for evaluation')
        # Subsample train and val splits for computing language generation during runtime
        
        if not self.seq2seq:
            training_lengths = [min(sum([sum(masks) for masks in self.train_dataloader.dataset[idx]['attention_mask']]), self.max_seq_len) for idx in range(len(self.train_dataloader.dataset))]
            length_counts = Counter(training_lengths)
            probs = torch.tensor([length_counts[idx]/len(self.train_dataloader.dataset) for idx in range(self.max_seq_len+1)])
            assert probs[0] == 0, 'Can\'t have examples of length 0'
            self.length_categorical = torch.distributions.Categorical(probs=probs)
        
        # optimizer

        self.opt = optimizer.get_adamw_optimizer(self.diffusion.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)
        
        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps*self.num_devices,
            num_training_steps=train_num_steps*self.num_devices,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion, beta = ema_decay, update_every = ema_update_every, power=3/4)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.diffusion, self.bart_model, self.opt, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(self.diffusion, self.bart_model, self.opt, self.train_dataloader, lr_scheduler)
        self.data_iter = cycle(self.train_dataloader)
        self.val_iter = cycle(self.val_dataloader)
        self.reference_dict = {}

    def save(self, best=False):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.diffusion),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'scheduler': self.lr_scheduler.state_dict(),
            }
        
        if best:
            torch.save(data, str(self.results_folder / f'best_model.pt'))
        else:
            torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None, best=False, init_only=False, only_decoder=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device
        
        data_bart = torch.load(str(file_path / 'bart_model.pt'), map_location=device, weights_only=True)        
        model_bart = self.accelerator.unwrap_model(self.bart_model)     
        model_bart.load_state_dict(data_bart['model'], strict = False)
        
        if not only_decoder:
            if best:
                data = torch.load(str(file_path / 'best_model.pt'), map_location=device, weights_only=True)
            else:
                data = torch.load(str(file_path / 'model.pt'), map_location=device, weights_only=True)
                
            model = self.accelerator.unwrap_model(self.diffusion)
            model.load_state_dict(data['model'], strict = False)
            
            self.opt.load_state_dict(data['opt'])
            if self.accelerator.is_local_main_process:
                self.ema.load_state_dict(data['ema'], strict = False)
            if init_only:
                return
            self.step = data['step']
            
            if 'scheduler' in data:
                self.lr_scheduler.load_state_dict(data['scheduler'])
                
            # For backwards compatibility with earlier models
            
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

    def log_reference_metrics(self, test=False):
        accelerator = self.accelerator
        if test:
            train_subset = ['|'.join(texts) for texts in self.train_data['text']][:self.num_samples]
            train_subset2 = ['|'.join(texts) for texts in self.train_data['text']][self.num_samples:(2*self.num_samples)] 
            test_subset = ['|'.join(texts) for texts in self.val_data['text']][:self.num_samples]

            self.reference_dict['reference/test_perplexity'] = evaluation.compute_perplexity(test_subset)
            for mauve_model_id in ["gpt2-large"]:
                self.reference_dict[f'reference/{mauve_model_id}_train_test_mauve'], _ = evaluation.compute_mauve(train_subset, test_subset, mauve_model_id)
                self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
                ngram_metrics = evaluation.compute_diversity(self.args, test_subset)
            for k, v in ngram_metrics.items():
                self.reference_dict[f"reference/test_{k}"] = v
            self.reference_dict[f"reference/test_memorization"] = evaluation.compute_memorization(self.args, test_subset, self.train_data)
            self.reference_dict['reference/test_unique_wordcount'] = evaluation.compute_wordcount(test_subset)
            return

        val_subset = ['|'.join(texts) for texts in self.val_data['text']][:self.num_samples]
        train_subset = ['|'.join(texts) for texts in self.train_data['text']][:self.num_samples]
        train_subset2 = ['|'.join(texts) for texts in self.train_data['text']][self.num_samples:(2*self.num_samples)]

        self.reference_dict['reference/train_perplexity'] = evaluation.compute_perplexity(train_subset)
        self.reference_dict['reference/val_perplexity'] = evaluation.compute_perplexity(val_subset)
        for mauve_model_id in ["gpt2-large"]:
            self.reference_dict[f'reference/{mauve_model_id}_train_val_mauve'], _ = evaluation.compute_mauve(train_subset, val_subset, mauve_model_id)
            self.reference_dict[f'reference/{mauve_model_id}_train_train_mauve'], _ = evaluation.compute_mauve(train_subset, train_subset2, mauve_model_id)
        ngram_metrics = evaluation.compute_diversity(self.args, val_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/val_{k}"] = v
        ngram_metrics = evaluation.compute_diversity(self.args, train_subset)
        for k, v in ngram_metrics.items():
            self.reference_dict[f"reference/train_{k}"] = v
        self.reference_dict[f"reference/val_memorization"] = evaluation.compute_memorization(self.args, val_subset, self.train_data)
        self.reference_dict['reference/train_unique_wordcount'] = evaluation.compute_wordcount(train_subset)
        self.reference_dict['reference/val_unique_wordcounts'] = evaluation.compute_wordcount(val_subset)
        torch.cuda.empty_cache() 
            
    @torch.no_grad()
    def sample(self, num_samples=None, class_id=None, seed=42, test=False, cls_free_guidance=1.0, infill=False, outfill=False):
        start_time = time.time()
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        self.diffusion.to('cpu')
        self.bart_model.to(device)
        torch.cuda.empty_cache() 

        self.ema.ema_model.eval()

        # Extract references
        reference_texts = {}        
        if test:
            reference_texts['test'] = ['|'.join(texts) for texts in self.val_data['text']][:num_samples]
            reference_texts['train'] = ['|'.join(texts) for texts in self.train_data['text']][:num_samples]
        else:
            reference_texts['val'] = ['|'.join(texts) for texts in self.val_data['text']][:num_samples]
            reference_texts['train'] = ['|'.join(texts) for texts in self.train_data['text']][:num_samples]

        milestone = self.step // self.save_and_sample_every
        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in constant.generate_kwargs.items()}    

        torch.manual_seed(seed)
        def get_class_id(n):
            if exists(class_id):
                return torch.tensor([class_id]*n, dtype=torch.long, device=device)
            if self.class_conditional:
                if self.diffusion.diffusion_model.class_unconditional_prob > 0:
                    return torch.tensor([self.diffusion.diffusion_model.num_classes]*n, dtype=torch.long, device=device)
                return self.class_categorical.sample((n,)).to(device)
            return None
        if (infill or outfill):
            val_dataset = MelodyDataset(self.args, self.val_data)
            dataloader = get_dataloader(self.args, val_dataset, self.bart_model.config, shuffle=False)
            data_iter = cycle(dataloader)
        # Loop until enough senetences have been generated across all strategies 
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            batches = num_to_groups(num_samples-min([len(all_texts_lists[ele]) for ele in all_texts_lists]), max(self.eval_batch_size,self.train_batch_size))
            if (infill or outfill):
                data = next(data_iter)
                for key, value in data.items():
                    data[key] = data[key].to(device)
                with torch.no_grad():
                    z_t = None
                    for i in range(len(data['input_ids'][0])):
                        encoder_outputs = self.bart_model.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                        if self.using_latent_model:
                            latent = self.bart_model.get_diffusion_latent(encoder_outputs, data['attention_mask'][:,i])      
                        else:                      
                            latent = encoder_outputs.last_hidden_state
                            
                        if self.args.normalize_latent:
                            latent = self.diffusion.normalize_latent(latent)
                        if z_t == None:
                            z_t = latent.clone()
                        else:
                            assert z_t.shape[0] == latent.shape[0] and z_t.shape[2] == latent.shape[2]
                            z_t = torch.cat((z_t, latent), dim=1)
                model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=[sum(data['attention_mask'][i]) for i in range(n)],
                                        class_id=get_class_id(n), cls_free_guidance=cls_free_guidance, z_t = z_t[:n].clone(), infill=infill, outfill=outfill)), batches))
            else:
                model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=get_class_id(n), cls_free_guidance=cls_free_guidance)), batches))
            
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)

                for k, kwargs in constant.generate_kwargs.items():
                    kwargs['max_length'] = self.max_seq_len
                    texts_list = []
                    for i in range(self.args.seg):
                        if self.latent_model_path:
                            attention_mask = None
                            encoder_output = BaseModelOutput(last_hidden_state=self.bart_model.get_decoder_input(latents[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone()))
                        else:
                            attention_mask = mask[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone()
                            encoder_output = BaseModelOutput(last_hidden_state=latents[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone())
                        sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **kwargs)
                        if texts_list == []:
                            texts_list = [''.join(chr(idx) if idx > 3 else '' for idx in g if idx != 3) for g in sample_ids]
                        else:
                            assert len(texts_list) == len(sample_ids)
                            for j in range(len(texts_list)):
                                texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > 3 else '' for idx in sample_ids[j] if idx != 3)}"
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    all_texts_lists[k].extend(texts_list)
        
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples
        text_generations = {k:v[:num_samples] for k,v in all_texts_lists.items()} 
        end_time = time.time()
        inference_time = end_time - start_time
        metrics = {}

        self.ema.to('cpu')
        torch.cuda.empty_cache() 
        for strategy, all_texts_list in text_generations.items():
            class_id_prefix = f'cond{class_id}_' if exists(class_id) else ''
            file_utils.save_text_samples(all_texts_list, os.path.join(self.results_folder, f'{"eval-" if self.args.eval else ""}{f"eval{seed}-" if self.args.eval_test else ""}{class_id_prefix}{strategy}-sample-{milestone}.txt'))
            metrics[f"model/{strategy}/{class_id_prefix}perplexity"] = evaluation.compute_perplexity(all_texts_list)
            metrics[f"model/{strategy}/{class_id_prefix}unique_wordcount"] = evaluation.compute_wordcount(all_texts_list)
            ngram_metrics = evaluation.compute_diversity(self.args, all_texts_list)
            for k, v in ngram_metrics.items():
                metrics[f"model/{strategy}/{class_id_prefix}{k}"] = v
            metrics[f"model/{strategy}/{class_id_prefix}memorization"] = evaluation.compute_memorization(self.args, all_texts_list, self.train_data)
            table = wandb.Table( 
                columns=['Samples'], data=[[text] for text in all_texts_list])
            accelerator.log({f"model/{strategy}/{class_id_prefix}samples": table}, self.step)

            # Only evaluate MAUVE if generations are reasonable to speed up validation early on
            if metrics[f"model/{strategy}/{class_id_prefix}perplexity"] > 5000:
                continue

            for mauve_model_id in ["gpt2-large"]:
                for key, reference_text in reference_texts.items():
                    metrics[f"model/{strategy}/{mauve_model_id}_{class_id_prefix}{key}_mauve"], _ = evaluation.compute_mauve(all_texts_list, reference_text, mauve_model_id)

        if len(self.reference_dict) == 0 or test:
            self.log_reference_metrics(test)
        if test:
            metrics_dict = {**metrics,**self.reference_dict}
            metrics_dict = {f'{k}_seed{seed}':v for k,v in metrics_dict.items()}
            accelerator.log(metrics_dict, self.step)
            print(metrics_dict)
        else:
            accelerator.log({**metrics,**self.reference_dict}, self.step)            
        print("metrics: ", metrics)

        eval_type = 'test' if test else 'val'
        with open( f'{self.args.save_dir}/{seed}_{eval_type}_reference.json', "w") as file:
            json.dump(reference_texts, file, ensure_ascii=False)
        print(f'prediction saved at ~/{seed}_{eval_type}_reference.json')
        start_time = time.time()
        with open( f'{self.args.save_dir}/{seed}_{eval_type}_prediction.json', "w") as file:
            json.dump(all_texts_lists, file, ensure_ascii=False)
        #print(f'prediction saved at ~/{seed}_{eval_type}_prediction.json')
        end_time = time.time()
        print(f'Convert time : {inference_time + end_time - start_time} second')

        torch.cuda.empty_cache() 
        self.diffusion.to(device)
        self.bart_model.to(device)
        self.ema.to(device)

    @torch.no_grad()
    def sample_seq2seq(self, num_samples=None, split='val', seed=42, num_candidates=None, cls_free_guidance=1.0,):
        assert split in ['train', 'val', 'test']
        num_samples = default(num_samples, self.num_samples) if split != 'test' else min(len(self.val_data['text']), self.num_samples)
        num_candidates = default(num_candidates, self.seq2seq_candidates)
        accelerator = self.accelerator
        device = accelerator.device

        self.ema.ema_model.eval()
        self.ema.to(device)

        # Extract references
        reference_texts = []
        source_texts = []
        pred_texts = []

        torch.manual_seed(seed)

        if split == 'val':
            dataloader = self.val_dataloader
            prefix = ''
        elif split == 'train':
            dataloader = self.train_val_dataloader
            prefix = 'train/'
        elif split == 'test':
            dataloader = self.test_dataloader
            prefix = 'test/'
        else:
            raise ValueError(f'invalid split {split}')
        
        diffusion = accelerator.unwrap_model(self.diffusion)
        prefix += f'guide{cls_free_guidance}/' if cls_free_guidance != 1.0 else ''
        
        if self.args.normalize_latent:
            self.ema.ema_model.seq2seq_latent_mean.to(device)
            self.ema.ema_model.seq2seq_latent_scale.to(device)
                
        for batch in dataloader:
            data = batch
            for key, value in data.items():
                data[key] = data[key].to(device)
            seq2seq_cond = diffusion.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
            seq2seq_mask = data['cond_attention_mask'].bool()
            if self.args.normalize_latent:
                seq2seq_cond = self.ema.ema_model.seq2seq_normalize_latent(seq2seq_cond)
            pred_cand_list = []
            ref_cand_list = []
            source_cand_list = []
            gen_kwargs = constant.generate_kwargs['beam']
            gen_kwargs['max_length'] = self.max_seq_len
            for _ in range(num_candidates):
                l2_normalize = self.diffusion.l2_normalize
                latents, mask = self.ema.ema_model.sample(batch_size=seq2seq_cond.shape[0], length=None, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask,
                                                            cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
                #latents, mask = self.ema.ema_model.sample(batch_size=seq2seq_cond.shape[0], length=self.length_categorical.sample((seq2seq_cond.shape[0],)), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
                texts_list = []
                data_list = []
                for i in range(self.args.seg):
                    if self.latent_model_path:
                        attention_mask = None
                        encoder_output = BaseModelOutput(last_hidden_state=self.bart_model.get_decoder_input(latents[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone()))
                    else:
                        attention_mask = mask[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone()
                        encoder_output = BaseModelOutput(last_hidden_state=latents[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone())
                    sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **gen_kwargs)
                    if texts_list == []:
                        texts_list = [''.join(chr(idx) if idx > 3 else '' for idx in g if idx != 3) for g in sample_ids]
                    else:
                        assert len(texts_list) == len(sample_ids)
                        for j in range(len(texts_list)):
                            texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > 3 else '' for idx in sample_ids[j] if idx != 3)}"
                texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                pred_cand_list.append(texts_list)
                for datas in data['input_ids'].cpu().numpy().tolist():
                    ref_list = []
                    for texts in datas:
                        ref_list.extend(texts)
                        ref_list.append(ord('|'))
                    ref_list = ref_list[:-1]
                    data_list.append(''.join(chr(idx) if idx > 3 else ''  for idx in ref_list if idx != 3))
                data_list = [text.strip() for text in data_list if len(text.strip())>0]
                ref_cand_list.append(data_list)
                source_cand_list.append([''.join(chr(idx) if idx > 3 else '' for idx in g if idx != 3) for g in data['cond_input_ids']])
            assert len(pred_cand_list) == num_candidates
            assert len(ref_cand_list) == num_candidates
            assert len(source_cand_list) == num_candidates
            pred_texts.extend([val for tup in zip(*pred_cand_list) for val in tup])
            reference_texts.extend([val for tup in zip(*ref_cand_list) for val in tup])
            source_texts.extend([val for tup in zip(*source_cand_list) for val in tup])
            if len(pred_texts) >= num_samples*num_candidates:
                break
        assert len(pred_texts) == len(reference_texts) == len(source_texts)
        assert len(pred_texts) >= num_samples*num_candidates
        pred_texts = pred_texts[:num_samples*num_candidates]
        reference_texts = reference_texts[:num_samples*num_candidates]
        source_texts = source_texts[:num_samples*num_candidates]

         # Save samples and references to json
        if split == 'test':
            samples_dict = {'pred_texts': pred_texts, 'reference_texts': reference_texts, 'source_texts': source_texts}
            save_path = os.path.join(self.results_folder, f'{prefix}_seq2seq_{split}_samples.json')    
            # Create dir if it doesn't exist   
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            with open(os.path.join(save_path), 'w') as f:
                json.dump(samples_dict, f)

        # Log samples
        # source | reference | pred
        columns = ['source', 'reference', 'pred']
        data = []
        for i in range(len(reference_texts)):
            row = [source_texts[i], reference_texts[i], pred_texts[i]]
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"seq2seq/{prefix}{split}_samples": table}, self.step)

        # Compute metrics
        metrics = {}

        # Get oracle rouge
        raw_rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts, use_aggregator=False)
        # Compute the max rouge score across num_candidates
        for k, v in raw_rouge_metrics.items():
            np_metric = np.array(v).reshape(num_samples, num_candidates)
            np_metric = np.max(np_metric, axis=1)
            metrics[f"model/seq2seq/{prefix}oracle_{k}"] = np_metric.mean().item()

        if num_candidates > 1:
            mbr_rouge_scores = np.zeros((num_samples, num_candidates))
            for i in range(num_candidates):
                pred_texts_i = pred_texts[i::num_candidates]
                for j in range(num_candidates):
                    if j == i:
                        continue
                    ref_texts_j = pred_texts[j::num_candidates]
                    rouge2_arr = np.array(evaluation.compute_rouge(pred_texts_i, ref_texts_j, use_aggregator=False)['rouge2'])
                    mbr_rouge_scores[:, i] += rouge2_arr
            best_indices = np.argmax(mbr_rouge_scores, axis=1)
            best_predictions = [pred_texts[i*num_candidates + idx] for i, idx in enumerate(best_indices)]
            mbr_rouge_metrics = evaluation.compute_rouge(best_predictions, reference_texts[::num_candidates])
            for k, v in mbr_rouge_metrics.items():
                metrics[f"model/seq2seq/{prefix}mbr_{k}"] = v
            metrics[f'model/seq2seq/{prefix}mbr_bertscore'] = evaluation.compute_bertscore(best_predictions, reference_texts[::num_candidates])

        # Get every num_candidates samples
        pred_texts = pred_texts[::num_candidates]
        reference_texts = reference_texts[::num_candidates]
        source_texts = source_texts[::num_candidates]
        
        rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts)
        for k, v in rouge_metrics.items():
            metrics[f"model/seq2seq/{prefix}{k}"] = v

        if rouge_metrics['rougeL'] > self.best_seq2seq_metric and split == 'val':
            self.best_seq2seq_metric = rouge_metrics['rougeL']
            self.save(best=True)

        rouge_metrics = evaluation.compute_rouge(pred_texts, reference_texts, use_stemmer=True)
        for k, v in rouge_metrics.items():
            metrics[f"model/seq2seq/{prefix}stem_{k}"] = v

        shuffled_pred_texts = random.sample(pred_texts, len(pred_texts))
        shuffled_rouge_metrics = evaluation.compute_rouge(shuffled_pred_texts, reference_texts)
        for k, v in shuffled_rouge_metrics.items():
            metrics[f"model/seq2seq/{prefix}shuffled_{k}"] = v

        #metrics[f"model/seq2seq/{prefix}perplexity"] = evaluation.compute_perplexity(pred_texts)
        metrics[f"model/seq2seq/{prefix}unique_wordcount"] = evaluation.compute_wordcount(pred_texts)
        ngram_metrics = evaluation.compute_diversity(self.args, pred_texts)
        for k, v in ngram_metrics.items():
            metrics[f"model/seq2seq/{prefix}{k}"] = v
        metrics[f"model/seq2seq/{prefix}memorization"] = evaluation.compute_memorization(self.args, pred_texts, self.train_data)
        metrics[f"model/seq2seq/{prefix}bertscore"] = evaluation.compute_bertscore(pred_texts, reference_texts)
        
        accelerator.log(metrics, self.step)
        print(metrics)
        prefix = prefix.replace("/", "_")
        with open( f'{self.args.save_dir}/{seed}_{prefix}_{split}_seq_seq_source.json', "w") as file:
            json.dump(source_texts, file, ensure_ascii=False)
        print(f'source saved at ~/{seed}_{prefix}_{split}_seq_seq_source.json')
        with open( f'{self.args.save_dir}/{seed}_{prefix}_{split}_seq_seq_prediction.json', "w") as file:
            json.dump(pred_texts, file, ensure_ascii=False)
        print(f'prediction saved at ~/{seed}_{prefix}_{split}_seq_seq_prediction.json')

        torch.cuda.empty_cache()

    @torch.no_grad()
    def inference(self, num_samples=None, seed=42, cls_free_guidance=1.0, infill=False, outfill=False):
        start_time = time.time()
        num_samples = default(num_samples, self.num_samples)
        accelerator = self.accelerator
        device = accelerator.device
        self.diffusion.to('cpu')
        self.bart_model.to(device)
        torch.cuda.empty_cache() 

        self.ema.ema_model.eval()

        # Stores generation outputs for each strategy
        all_texts_lists = {k:[] for k,_ in constant.generate_kwargs.items()}    

        torch.manual_seed(seed)
        
        if (infill or outfill):
            val_dataset = MelodyDataset(self.args, self.val_data)
            dataloader = get_dataloader(self.args, val_dataset, self.bart_model.config, shuffle=False)
            data_iter = cycle(dataloader)
        # Loop until enough senetences have been generated across all strategies 
        while min([len(all_texts_lists[ele]) for ele in all_texts_lists]) < num_samples:
            batches = num_to_groups(num_samples-min([len(all_texts_lists[ele]) for ele in all_texts_lists]), max(self.eval_batch_size,self.train_batch_size))
            if (infill or outfill):
                data = next(data_iter)
                for key, value in data.items():
                    data[key] = data[key].to(device)
                with torch.no_grad():
                    z_t = None
                    for i in range(len(data['input_ids'][0])):
                        encoder_outputs = self.bart_model.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                        if self.using_latent_model:
                            latent = self.bart_model.get_diffusion_latent(encoder_outputs, data['attention_mask'][:,i])      
                        else:                      
                            latent = encoder_outputs.last_hidden_state
                            
                        if self.args.normalize_latent:
                            latent = self.diffusion.normalize_latent(latent)
                        if z_t == None:
                            z_t = latent.clone()
                        else:
                            assert z_t.shape[0] == latent.shape[0] and z_t.shape[2] == latent.shape[2]
                            z_t = torch.cat((z_t, latent), dim=1)
                model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=[sum(data['attention_mask'][i]) for i in range(n)],
                                        class_id=None, cls_free_guidance=cls_free_guidance, z_t = z_t[:n].clone(), infill=infill, outfill=outfill)), batches))
            else:
                model_outputs = list(map(lambda n: tuple(x.to('cpu') for x in self.ema.ema_model.sample(batch_size=n, length=self.length_categorical.sample((n,)), class_id=None, cls_free_guidance=cls_free_guidance)), batches))
            
            for (latents, mask) in model_outputs:
                latents, mask = latents.to(device), mask.to(device)

                for k, kwargs in constant.generate_kwargs.items():
                    kwargs['max_length'] = self.max_seq_len
                    texts_list = []
                    for i in range(self.args.seg):
                        if self.latent_model_path:
                            attention_mask = None
                            encoder_output = BaseModelOutput(last_hidden_state=self.bart_model.get_decoder_input(latents[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone()))
                        else:
                            attention_mask = mask[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone()
                            encoder_output = BaseModelOutput(last_hidden_state=latents[:,i*self.num_encoder_latents:(i+1)*self.num_encoder_latents].clone())
                        sample_ids = self.bart_model.generate(encoder_outputs=encoder_output, attention_mask=attention_mask, **kwargs)
                        if texts_list == []:
                            texts_list = [''.join(chr(idx) if idx > 3 else '' for idx in g if idx != 3) for g in sample_ids]
                        else:
                            assert len(texts_list) == len(sample_ids)
                            for j in range(len(texts_list)):
                                texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > 3 else '' for idx in sample_ids[j] if idx != 3)}"
                    texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                    all_texts_lists[k].extend(texts_list)
        
        assert min([len(all_texts_lists[ele]) for ele in all_texts_lists]) >= num_samples

        self.ema.to('cpu')
        torch.cuda.empty_cache()

        for key, value in all_texts_lists.items():
            os.makedirs(os.path.join(self.args.save_dir, key), exist_ok = True)
        for key, value in all_texts_lists.items():
            for i, data in enumerate(value):
                # Your ABC notation data
                abc_data = [
                    "L:1/8",
                    "M:4/4"
                ]
                abc_data.append(data)
                # Specify the output file path
                abc_file_path = os.path.join(self.args.save_dir, f'{key}/{i}.abc')
                abc_file_path = abc_file_path.replace('\\', '/')

                # Write the ABC data to the file
                with open(abc_file_path, 'w') as abc_file:
                    for line in abc_data:
                        abc_file.write(line + '\n')
                midi_file = os.path.join(self.args.save_dir, f'{key}/{i}.mid')
                midi_file = midi_file.replace('\\', '/')
                try:
                    score = converter.parse(abc_file_path)
                    mf = midi.translate.music21ObjectToMidiFile(score)
                    mf.open(midi_file, 'wb')
                    mf.write()
                    mf.close()
                except Exception as error:            
                    try:
                        score = converter.parse(abc_file_path)
                        new_score = stream.Score()
                        for part in score.parts:
                            new_part = stream.Part()
                            existing_time_signatures = set()
                            for element in part.flat.notesAndRests:
                                if isinstance(element, meter.TimeSignature):
                                    if element not in existing_time_signatures:
                                        existing_time_signatures.add(element)
                                        new_part.append(element)
                                else:
                                    new_part.append(element)
                            new_score.append(new_part)
                        mf = midi.translate.music21ObjectToMidiFile(new_score)
                        mf.open(midi_file, 'wb')
                        mf.write()
                        mf.close()
                    except Exception as error:
                        pass
        end_time = time.time()
        print(f'Convert time : {end_time - start_time} second')

        with open( f'{self.args.save_dir}/{seed}_val_prediction.json', "w") as file:
            json.dump(all_texts_lists, file, ensure_ascii=False)
        print(f'prediction saved at ~/{seed}_val_prediction.json')
        torch.cuda.empty_cache() 
        self.diffusion.to(device)
        self.bart_model.to(device)
        self.ema.to(device)

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            if self.step > 0:
                data = next(islice(self.data_iter, self.step-1, None))
                if self.step >= 500:
                    val_data = next(islice(self.val_iter, (self.step//500)-1, None))
                for i in range(self.step + (self.step//250)):
                    torch.empty((self.train_batch_size, self.max_seq_len, self.diffusion.latent_dim), dtype=torch.float, device=device).normal_(0, 1)
                    torch.empty((self.train_batch_size), dtype=torch.float, device=device).normal_(0, 1)

                    if self.diffusion.self_condition:
                        random.random()
                    if self.seq2seq:
                        random.random()
                
            if self.args.normalize_latent:
                self.diffusion.latent_mean.to(device)
                self.diffusion.latent_scale.to(device)
                self.ema.ema_model.latent_mean = self.diffusion.latent_mean
                self.ema.ema_model.latent_scale = self.diffusion.latent_scale
                if self.seq2seq:
                    self.diffusion.seq2seq_latent_mean.to(device)
                    self.diffusion.seq2seq_latent_scale.to(device)
                    self.ema.ema_model.seq2seq_latent_mean = self.diffusion.seq2seq_latent_mean
                    self.ema.ema_model.seq2seq_latent_scale = self.diffusion.seq2seq_latent_scale
                    
            accelerator.wait_for_everyone()
                
            while self.step < self.train_num_steps:

                #TODO center and normalize BART latent space with empirical est. of mean/var.

                total_loss = 0.
                decoding_loss = 0.
                for grad_accum_step in range(self.gradient_accumulate_every):
                    data = next(self.data_iter)
                    for key, value in data.items():
                        data[key] = data[key].to(device)
                    with torch.no_grad():
                        latents = None
                        for i in range(len(data['input_ids'][0])):
                            encoder_outputs = self.bart_model.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                            if self.using_latent_model:
                                latent = self.bart_model.get_diffusion_latent(encoder_outputs, data['attention_mask'][:,i])      
                            else:                      
                                latent = encoder_outputs.last_hidden_state
                            
                            if self.args.normalize_latent:
                                latent = self.diffusion.normalize_latent(latent)
                            if latents == None:
                                latents = latent.clone()
                            else:
                                assert latents.shape[0] == latent.shape[0] and latents.shape[2] == latent.shape[2]
                                latents = torch.cat((latents, latent), dim=1)
                    seq2seq_cond = None
                    seq2seq_mask = None
                    with accelerator.autocast():
                        if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                            if self.num_devices > 1:
                                seq2seq_cond = self.diffusion.module.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                            else:
                                seq2seq_cond = self.diffusion.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                            seq2seq_mask = data['cond_attention_mask'].bool()
                            if self.args.normalize_latent:
                                seq2seq_cond = self.diffusion.seq2seq_normalize_latent(seq2seq_cond)

                    if self.using_latent_model:
                        mask = torch.ones(latents.shape[0], self.num_encoder_latents * self.args.seg, dtype=torch.bool).to(device)
                    else:
                        x, y, z = data['attention_mask'].shape
                        mask = data['attention_mask'].view(x, y * z).bool()
                    if self.decoding_loss:
                        raise NotImplementedError
                    else:
                        loss = self.diffusion(latents, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        
                    self.accelerator.backward(loss) 

                accelerator.clip_grad_norm_(self.diffusion.parameters(), self.args.clip_grad_norm)
                grad_norm = compute_grad_norm(self.diffusion.parameters())

                accelerator.wait_for_everyone()
                self.opt.step()
                self.lr_scheduler.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    logs = {
                        "loss": total_loss,
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm,
                        "step": self.step, 
                        "epoch": (self.step*self.gradient_accumulate_every)/len(self.train_dataloader), 
                        "samples": self.step*self.train_batch_size*self.gradient_accumulate_every*self.num_devices
                    }
                    if self.decoding_loss:
                        logs['decoding_loss'] = decoding_loss
                    self.ema.to(device)
                    self.ema.update()

                    # Log to WandB
                    if self.step % 500 == 0:
                        self.diffusion.eval()
                        self.ema.ema_model.eval()
                        with torch.no_grad():
                            total_val_loss = 0.
                            total_val_ema_loss = 0.
                            for grad_accum_step in range(self.gradient_accumulate_every):
                                data = next(self.val_iter)
                                for key, value in data.items():
                                    data[key] = data[key].to(device)
                                latents = None
                                for i in range(len(data['input_ids'][0])):               
                                    encoder_outputs = self.bart_model.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                                    if self.using_latent_model:
                                        latent = self.bart_model.get_diffusion_latent(encoder_outputs, data['attention_mask'][:,i])      
                                    else:                      
                                        latent = encoder_outputs.last_hidden_state
                                    
                                    if self.args.normalize_latent:
                                        latent = self.diffusion.normalize_latent(latent)
                                    if latents == None:
                                        latents = latent.clone()
                                    else:
                                        assert latents.shape[0] == latent.shape[0] and latents.shape[2] == latent.shape[2]
                                        latents = torch.cat((latents, latent), dim=1)
                                seq2seq_cond = None
                                seq2seq_mask = None
                                if self.seq2seq and random.random() < (1-self.seq2seq_unconditional_prob):
                                    with torch.no_grad():
                                        if self.num_devices > 1:
                                            seq2seq_cond = self.diffusion.module.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                                        else:
                                            seq2seq_cond = self.diffusion.context_encoder(input_ids = data['cond_input_ids'], attention_mask = data['cond_attention_mask']).last_hidden_state.float()
                                    seq2seq_mask = data['cond_attention_mask'].bool()
                                    if self.args.normalize_latent:
                                        seq2seq_cond = self.diffusion.seq2seq_normalize_latent(seq2seq_cond)
                                
                                if self.using_latent_model:
                                    mask = torch.ones((latents.shape[0], self.num_encoder_latents * self.args.seg), dtype=torch.bool).to(device)
                                else:
                                    x, y, z = data['attention_mask'].shape
                                    mask = data['attention_mask'].view(x, y * z).bool()
                                loss = self.diffusion(latents, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                                loss = loss / self.gradient_accumulate_every
                                total_val_loss += loss.item()
                                loss = self.ema.ema_model(latents, mask, class_id=(data['label'] if self.class_conditional else None), seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                                loss = loss / self.gradient_accumulate_every
                                total_val_ema_loss += loss.item()

                            logs["val_loss"] = total_val_loss 
                            logs["val_ema_loss"] = total_val_ema_loss
                            pbar.set_postfix(**logs)
                        
                        print("logs : ", logs)
                        self.diffusion.train()

                    accelerator.log(logs, step=self.step)              
                    if self.step % self.save_and_sample_every == 0:
                        self.save()
                        if self.seq2seq:
                            self.sample_seq2seq()
                            self.sample_seq2seq(split='train')
                        else:
                            self.sample()
                        if self.class_conditional:
                            for class_id in range(self.diffusion.diffusion_model.num_classes):
                                self.sample(num_samples=100, class_id=class_id)                        
                        
                        self.diffusion.train()
                pbar.update(1)
            accelerator.wait_for_everyone()
        self.save()
        accelerator.print('training complete')