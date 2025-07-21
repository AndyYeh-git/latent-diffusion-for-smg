# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 21:26:01 2024

@author: User
"""

import math
from functools import partial

import os
import numpy as np

import argparse
from datetime import timedelta

import torch
import torch.nn as nn

from transformers import get_scheduler, BertGenerationConfig, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, BartConfig, BartForConditionalGeneration, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import diffusion.optimizer as optimizer
from dataset_utils.create_dataloader import get_dataset, MelodyDataset, get_dataloader
import diffusion.constant as constant

from evaluation import evaluation
from pathlib import Path
import copy
import random
from tqdm.auto import tqdm

from itertools import cycle, islice

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def exists(val):
    return val is not None

def normalize_latent(x_start, latent_mean, latent_scale):
    eps = 1e-5

    return (x_start-latent_mean)/(latent_scale).clamp(min=eps)

def seq2seq_normalize_latent(x_start, seq2seq_latent_mean, seq2seq_latent_scale):
    eps = 1e-5

    return (x_start-seq2seq_latent_mean)/(seq2seq_latent_scale).clamp(min=eps)

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

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

def random_masking(args, model, data, count1, count2):
    for i in range(data['input_ids'].size()[0]):
        if 'flan-t5' in args.enc_dec_model:
            for j in range(data['input_ids'].size()[1]):
                for k in range((sum(data['attention_mask'][i][j])-1)):
                    count1 += 1
                    if random.random() < args.prob:
                        count1 += 1
                        if random.random() < 0.5:
                            data['input_ids'][i][j][k] = 3
                        else:
                            count2 += 1
                            data['input_ids'][i][j][k] = random.randint(4, 127)
        else:
            for j in range(data['input_ids'].size()[1]):
                for k in range(1, (sum(data['attention_mask'][i][j])-1)):
                    count1 += 1
                    if random.random() < args.prob:
                        count1 += 1
                        if random.random() < 0.5:
                            data['input_ids'][i][j][k] = 3
                        else:
                            count2 += 1
                            data['input_ids'][i][j][k] = random.randint(4, 127)

    return data, count1, count2

def save(args, step, accelerator, model, opt, lr_scheduler, count1, count2):
    data = {
        'step': step,
        'model': accelerator.get_state_dict(model),
        'opt': opt.state_dict(),
        'scaler': accelerator.scaler.state_dict() if exists(accelerator.scaler) else None,
        'scheduler': lr_scheduler.state_dict(),
        'count1': count1,
        'count2': count2,
        }
        
    ck_path = os.path.join(args.save_dir, 'bart_model.pt')
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(data, ck_path)

def load(args, accelerator, model, opt, lr_scheduler, file_path=None):
    file_path = Path(file_path) if exists(file_path) else args.results_folder
    device = accelerator.device
        
    data = torch.load(str(file_path / 'bart_model.pt'), map_location=device, weights_only=True)        
    model = accelerator.unwrap_model(model)     
    model.load_state_dict(data['model'], strict = False)
        
    opt.load_state_dict(data['opt'])
    step = data['step']
    count1 = data['count1']
    count2 = data['count2']
            
    if 'scheduler' in data:
        lr_scheduler.load_state_dict(data['scheduler'])
                
    # For backwards compatibility with earlier models
            
    if exists(accelerator.scaler) and exists(data['scaler']):
        accelerator.scaler.load_state_dict(data['scaler'])
    
    return step, model, opt, lr_scheduler, count1, count2

def main(args):
    set_seeds(args.seed)
    if 'bart' in args.enc_dec_model:
        # Init Bart model Config
        config = BartConfig.from_pretrained(args.enc_dec_model)
        config.max_position_embeddings = max(args.max_seq_len, config.max_position_embeddings)
        config.vocab_size=args.vocab_size

        # Initialize the Bart model
        model = BartForConditionalGeneration(config = config)
        # load_model_pretrained_weights(args, model)
        model_type = 'bart'
    elif 'flan-t5' in args.enc_dec_model:
        config = T5Config.from_pretrained(args.enc_dec_model)
        config.vocab_size=args.vocab_size
        model = T5ForConditionalGeneration(config = config)
        model_type = 't5'
    elif 'bert' in args.enc_dec_model:
        # Init Encoder decoder Config
        config = BertGenerationConfig.from_pretrained(args.enc_dec_model)
        config.vocab_size=args.vocab_size
        # Init Encoder-decoder model
        encoder = BertGenerationEncoder(config = config)
        # add cross attention layers and use BERT's cls token as BOS token and sep token as EOS token
        config.is_decoder=True
        config.add_cross_attention=True
        decoder = BertGenerationDecoder(config = config)
        # Initialize the EncoderDecoderModel
        model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
        model.config.bos_token_id=model.encoder.config.bos_token_id
        model.config.eos_token_id=model.encoder.config.eos_token_id
        model.config.pad_token_id=model.encoder.config.pad_token_id
        model_type = 'bert'
    else:
        raise ValueError(f'invalid enc_dec_model {args.enc_dec_model}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if args.train_decoder:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_decoder().parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True
    skip_token_id = 3

    train_data, val_data = get_dataset(args)
    seq2seq = args.seq2seq

    train_dataset = MelodyDataset(args, train_data)
    val_dataset = MelodyDataset(args, val_data)
    
    dataloader = get_dataloader(args, train_dataset, model.config, shuffle=True)
    val_dataloader = get_dataloader(args, val_dataset, model.config, shuffle=True)

    if args.l2_normalize_latents:
        assert not args.normalize_latent

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
            if seq2seq:
                seq2seq_latent_mean = mean_std.split("[")[3].split("]\n")[0]
                seq2seq_latent_scale = mean_std.split("[")[4].split("]\n")[0]

        latent_mean = torch.tensor(np.fromstring(diffusion_latent_mean, dtype=float, sep=' ')).float()
        latent_scale = torch.tensor(np.fromstring(diffusion_latent_scale, dtype=float, sep=' ')).float()
        
        if seq2seq:
            seq2seq_latent_mean = torch.tensor(np.fromstring(seq2seq_latent_mean, dtype=float, sep=' ')).float()
            seq2seq_latent_scale = torch.tensor(np.fromstring(seq2seq_latent_scale, dtype=float, sep=' ')).float()
    
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    init_process_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=90))
    
    accelerator = Accelerator(
        mixed_precision = 'no',
        kwargs_handlers=[ddp_kwargs, init_process_kwargs]
    )
    
    opt = optimizer.get_adamw_optimizer(model.parameters(), lr = args.learning_rate, betas = (args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay)
    
    lr_scheduler = get_scheduler(
        args.lr_schedule,
        optimizer=opt,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_train_steps
    )
    
    scale = 1.
    step = 0
    count1 = 0    
    count2 = 0

    train_schedule = args.train_schedule
    
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
        self_train_schedule = alpha_schedule
    else:
        self_train_schedule = partial(time_to_alpha, alpha_schedule=alpha_schedule, scale=scale)
    
    model.train()
    model, opt, dataloader, lr_scheduler = accelerator.prepare(model, opt, dataloader, lr_scheduler)

    if args.eval:
        step, model, opt, lr_scheduler, count1, count2 = load(args, accelerator, model, opt, lr_scheduler, file_path=args.resume_dir)
        eval_dataloader = get_dataloader(args, val_dataset, model.config, shuffle=False)
        eval_iter = cycle(eval_dataloader)
        model.eval()
        data = next(eval_iter)
        if 'bart' in args.enc_dec_model:
            labels = data["labels"]
        else:
            data["decoder_input_ids"] = data["input_ids"].clone()
            data["decoder_input_ids"][:, :, 1:] = data["input_ids"][:, :, :-1]
            if 'flan-t5' in args.enc_dec_model:
                data["decoder_input_ids"][:, :, 0] = model.config.pad_token_id
            else:
                data["decoder_input_ids"][:, :, 0] = max(model.config.eos_token_id, model.config.bos_token_id) if model.config.bos_token_id != None else model.config.eos_token_id
            data["decoder_input_ids"].to(device)
            data["decoder_attention_mask"] = (data["decoder_input_ids"] != model.config.pad_token_id).long()
            data['labels'][data['labels'] == model.config.pad_token_id] = -100
            labels = data["labels"]

            assert labels.shape == data["decoder_input_ids"].shape
            assert data["decoder_attention_mask"].shape == data["decoder_input_ids"].shape
        for key, value in data.items():
            data[key] = data[key].to(device)
        with torch.no_grad():
            total_val_loss = 0.                        
            texts_list = []
            for i in range(len(data['input_ids'][0])):
                encoder_outputs = model.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                latents = encoder_outputs.last_hidden_state
                if args.normalize_latent:
                    latents = normalize_latent(latents, latent_mean.to(device), latent_scale.to(device))
                mask = None
                encoder_outputs = BaseModelOutput(last_hidden_state=latents.clone())
                loss = model(attention_mask = mask, decoder_input_ids = data["decoder_input_ids"][i], decoder_attention_mask = data["decoder_attention_mask"][i],
                            encoder_outputs = encoder_outputs, labels = labels[:,i].clone()).loss
                total_val_loss += loss.item()
                gen_kwargs = constant.generate_kwargs['beam']
                gen_kwargs['max_length'] = args.max_seq_len
                outputs = model.generate(attention_mask = mask, encoder_outputs = encoder_outputs, **gen_kwargs)

                if texts_list == []:
                    texts_list = [''.join(chr(idx) if idx > skip_token_id else '' for idx in g if idx != skip_token_id) for g in outputs]
                else:
                    assert len(texts_list) == len(outputs)
                    for j in range(len(texts_list)):
                        texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > skip_token_id else '' for idx in outputs[j] if idx != skip_token_id)}"
            print(f"total_val_loss : {total_val_loss / len(data['input_ids'][0])}") 
            texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
            data_list = []
            for datas in data['input_ids'].cpu().numpy().tolist():
                ref_list = []
                for texts in datas:
                    ref_list.extend(texts)
                    ref_list.append(ord('|'))
                ref_list = ref_list[:-1]
                data_list.append(''.join(chr(idx) if idx > skip_token_id else ''  for idx in ref_list if idx != skip_token_id))
                data_list = [text.strip() for text in data_list if len(text.strip())>0]
            print("data_list : ", data_list)
            print("texts_list : ", texts_list)
            if not seq2seq:
                try:
                    mauve_model_id = "gpt2-large"
                    mauve_results, _ = evaluation.compute_mauve(texts_list, data_list, mauve_model_id)
                    print(f'mauve_results : {mauve_results}')
                    bert_results = evaluation.compute_bertscore(texts_list, data_list)
                    print(f'bert_results : {bert_results}')
                except Exception as error:
                    print(f'Failed due to {type(error).__name__} - {error}')
            else:
                try:
                    bert_results = evaluation.compute_bertscore(texts_list, data_list)
                    print(f'bert_results : {bert_results}')
                except Exception as error:
                    print(f'Failed due to {type(error).__name__} - {error}')
        return

    if args.resume_training:
        step, model, opt, lr_scheduler, count1, count2 = load(args, accelerator, model, opt, lr_scheduler, file_path=args.resume_dir)
        model.train()    
    for i in range(count1):
        random.random()
    for i in range(count2):
        random.randint(4, 127)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(trainable_params)
    
    data_iter = cycle(dataloader)
    val_iter = cycle(val_dataloader)
    
    with tqdm(initial = step, total = args.num_train_steps, disable = not accelerator.is_main_process) as pbar:
        if step != 0:
            data = next(islice(data_iter, (step//args.seg)-1, None))
            if step >= args.sample_every:
                val_data = next(islice(val_iter, (step//args.sample_every)-1, None))    
        while step < args.num_train_steps:
            total_loss = 0.
            data = next(data_iter)
            if 'bart' in args.enc_dec_model:
                labels = data["labels"]
            else:
                data["decoder_input_ids"] = data["input_ids"].clone()
                data["decoder_input_ids"][:, :, 1:] = data["input_ids"][:, :, :-1]
                if 'flan-t5' in args.enc_dec_model:
                    data["decoder_input_ids"][:, :, 0] = model.config.pad_token_id
                else:
                    data["decoder_input_ids"][:, :, 0] = max(model.config.eos_token_id, model.config.bos_token_id) if model.config.bos_token_id != None else model.config.eos_token_id
                data["decoder_input_ids"].to(device)
                data["decoder_attention_mask"] = (data["decoder_input_ids"] != model.config.pad_token_id).long()
                data['labels'][data['labels'] == model.config.pad_token_id] = -100
                labels = data["labels"]

                assert labels.shape == data["decoder_input_ids"].shape
                assert data["decoder_attention_mask"].shape == data["decoder_input_ids"].shape
            for key, value in data.items():
                data[key] = data[key].to(device)
            new_data = copy.deepcopy(data)
            if args.random_masking:
                new_data, count1, count2 = random_masking(args, model, new_data, count1, count2)
            for i in range(len(new_data['input_ids'][0])):
                with torch.no_grad():
                    encoder_outputs = model.get_encoder()(input_ids = new_data['input_ids'][:,i], attention_mask = new_data['attention_mask'][:,i])
                    latents = encoder_outputs.last_hidden_state
                    if args.normalize_latent:
                        latents = normalize_latent(latents, latent_mean.to(device), latent_scale.to(device))
                    # mask = new_data['attention_mask'][:,i].bool()
                    mask = None
                
                    batch = latents.shape[0]
                    times = torch.zeros((batch,), device = device).float().uniform_(0, 0.15)

                    # noise sample
                    noise = torch.randn_like(latents)
                    alpha = self_train_schedule(times)
                    alpha = right_pad_dims_to(latents, alpha)
                    latents_D = alpha.sqrt() * latents + (1-alpha).sqrt() * noise

                encoder_outputs = BaseModelOutput(last_hidden_state=latents.clone())
                #encoder_outputs = BaseModelOutput(last_hidden_state=latents_D.clone())
                loss = model(attention_mask = mask, decoder_input_ids = new_data["decoder_input_ids"][i], decoder_attention_mask = new_data["decoder_attention_mask"][i],
                            encoder_outputs = encoder_outputs, labels = labels[:,i].clone()).loss
                total_loss += loss.item()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                accelerator.wait_for_everyone()
                opt.step()
                lr_scheduler.step()
                opt.zero_grad()
                accelerator.wait_for_everyone()
                step += 1

            if accelerator.is_main_process:
                logs = {
                        "loss": total_loss/len(new_data['input_ids'][0]),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "step": step,
                        "epoch": (step)/(len(dataloader)*len(new_data['input_ids'][0])),
                        "samples": (step*args.train_batch_size)/len(new_data['input_ids'][0])
                    }
                if step % 1000 == 0:
                    print(f"Step : {step}, total_loss : {total_loss / len(new_data['input_ids'][0])}")
                    pbar.set_postfix(**logs)
                if step % args.sample_every == 0:
                    model.eval()
                    data = next(val_iter)
                    if 'bart' in args.enc_dec_model:
                        labels = data["labels"]
                    else:
                        data["decoder_input_ids"] = data["input_ids"].clone()
                        data["decoder_input_ids"][:, :, 1:] = data["input_ids"][:, :, :-1]
                        if 'flan-t5' in args.enc_dec_model:
                            data["decoder_input_ids"][:, :, 0] = model.config.pad_token_id
                        else:
                            data["decoder_input_ids"][:, :, 0] = max(model.config.eos_token_id, model.config.bos_token_id) if model.config.bos_token_id != None else model.config.eos_token_id
                        data["decoder_input_ids"].to(device)
                        data["decoder_attention_mask"] = (data["decoder_input_ids"] != model.config.pad_token_id).long()
                        data['labels'][data['labels'] == model.config.pad_token_id] = -100
                        labels = data["labels"]

                        assert labels.shape == data["decoder_input_ids"].shape
                        assert data["decoder_attention_mask"].shape == data["decoder_input_ids"].shape
                    for key, value in data.items():
                        data[key] = data[key].to(device)
                    with torch.no_grad():
                        total_val_loss = 0.                        
                        texts_list = []
                        for i in range(len(data['input_ids'][0])):
                            encoder_outputs = model.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                            latents = encoder_outputs.last_hidden_state
                            if args.normalize_latent:
                                latents = normalize_latent(latents, latent_mean.to(device), latent_scale.to(device))
                            # mask = data['attention_mask'][:,i].bool()
                            mask = None
                            encoder_outputs = BaseModelOutput(last_hidden_state=latents.clone())
                            loss = model(attention_mask = mask, decoder_input_ids = data["decoder_input_ids"][i], decoder_attention_mask = data["decoder_attention_mask"][i],
                                        encoder_outputs = encoder_outputs, labels = labels[:,i].clone()).loss
                            total_val_loss += loss.item()
                            gen_kwargs = constant.generate_kwargs['beam']
                            gen_kwargs['max_length'] = args.max_seq_len
                            outputs = model.generate(attention_mask = mask, encoder_outputs = encoder_outputs, **gen_kwargs)

                            if texts_list == []:
                                texts_list = [''.join(chr(idx) if idx > skip_token_id else '' for idx in g if idx != skip_token_id) for g in outputs]
                            else:
                                assert len(texts_list) == len(outputs)
                                for j in range(len(texts_list)):
                                    texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > skip_token_id else '' for idx in outputs[j] if idx != skip_token_id)}"
                        logs["val_loss"] = total_val_loss / len(data['input_ids'][0])
                        print(f"total_val_loss : {total_val_loss / len(data['input_ids'][0])}") 
                        texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                        data_list = []
                        for datas in data['input_ids'].cpu().numpy().tolist():
                            ref_list = []
                            for texts in datas:
                                ref_list.extend(texts)
                                ref_list.append(ord('|'))
                            ref_list = ref_list[:-1]
                            data_list.append(''.join(chr(idx) if idx > skip_token_id else ''  for idx in ref_list if idx != skip_token_id))
                        data_list = [text.strip() for text in data_list if len(text.strip())>0]
                        print("data_list : ", data_list)
                        print("texts_list : ", texts_list)
                        if not seq2seq:
                            try:
                                mauve_model_id = "gpt2-large"
                                mauve_results, _ = evaluation.compute_mauve(texts_list, data_list, mauve_model_id)
                                print(f'mauve_results : {mauve_results}')
                                bert_results = evaluation.compute_bertscore(texts_list, data_list)
                                print(f'bert_results : {bert_results}')
                            except Exception as error:
                                print(f'Failed due to {type(error).__name__} - {error}')
                        else:
                            try:
                                bert_results = evaluation.compute_bertscore(texts_list, data_list)
                                print(f'bert_results : {bert_results}')
                            except Exception as error:
                                print(f'Failed due to {type(error).__name__} - {error}')
                        pbar.set_postfix(**logs)
                    print(f'logs : {logs}')
                    model.train()
                accelerator.log(logs, step=step)
                if step % args.save_every == 0:
                    save(args, step, accelerator, model, opt, lr_scheduler, count1, count2)
                    model.train()
            pbar.update(args.seg)
    save(args, step, accelerator, model, opt, lr_scheduler, count1, count2)
    accelerator.print('training complete')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default="irishman")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq2seq", action="store_true", default=False)
    parser.add_argument("--train_decoder", action="store_true", default=False)
    parser.add_argument("--save_dir", type=str, default="ck_save_path")
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--seg", type=int, default=4)
    parser.add_argument("--strip", action="store_true", default=False)
    parser.add_argument("--min", type=int, default=None)
    parser.add_argument("--L", type=str, default=None)
    parser.add_argument("--M", type=str, default=None)
    parser.add_argument("--random_masking", action="store_true", default=False)
    parser.add_argument("--prob", type=float, default=0.0)

    # Optimization hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--num_train_steps", type=int, default=250000)    
    parser.add_argument("--max_seq_len", type=int, default=32)
    parser.add_argument("--context_max_seq_len", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--l2_normalize_latents", action="store_true", default=False)
    parser.add_argument("--normalize_latent", action="store_true", default=False)

    # Model hyperparemeters
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument("--sample_every", type=int, default=2000)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument(
        "--train_schedule",
        type=str,
        default="cosine",
        choices=["beta_linear", "simple_linear", "cosine", 'sigmoid', 'tand', 'laplace'],
        help=(
            "Which noise schedule to use."
        ),
    )

    # Load and eval model
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--resume_training", action="store_true", default=False)
    parser.add_argument("--resume_dir", type=str, default=None)
    
    args = parser.parse_args()
    assert not (args.eval and args.resume_training)
    if args.eval or args.resume_training:
        assert args.resume_dir is not None
    
    main(args)