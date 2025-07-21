import math
import copy
from pathlib import Path
import random 
from functools import partial
from collections import namedtuple, Counter
from multiprocessing import cpu_count
import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from contextlib import nullcontext
import json

import torch
from torch import nn, einsum
import torch.nn.functional as F

import timeit

from einops import rearrange, reduce, repeat

from PIL import Image
from tqdm.auto import tqdm

from transformers import get_scheduler, T5ForConditionalGeneration, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartForConditionalGeneration
from datasets import concatenate_datasets

from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb

import CONSTANTS as CONSTANTS
import diffusion.optimizer as optimizer
from dataset_utils.create_dataloader import get_dataset, MelodyDataset, get_dataloader
from utils.torch_utils import compute_grad_norm
import utils.file_utils as file_utils
from evaluation import evaluation

from latent_models.bart_latent_model import BARTForConditionalGenerationLatent
from latent_models.t5_latent_model import T5ForConditionalGenerationLatent
from latent_models.latent_utils import get_latent_model

from itertools import cycle, islice

generate_kwargs = {
    'beam': 
    {'max_length':64, 'min_length':5, 'do_sample':False, 'num_beams':4, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2},
    'nucleus':
    {'max_length':64, 'min_length':5, 'do_sample':True, 'top_p':.95, 'num_beams':1, 'no_repeat_ngram_size':3, 'repetition_penalty':1.2}}

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# def cycle(dl):
#     while True:
#         for data in dl:
#             yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# trainer class

class Trainer(object):
    def __init__(
        self,
        args,
        dataset_name,
        *,
        train_batch_size = 16,
        eval_batch_size = 64,
        train_lr = 1e-4,
        train_num_steps = 100000,
        lr_schedule = 'cosine',
        num_warmup_steps = 500,
        adam_betas = (0.9, 0.99),
        adam_weight_decay = 0.01,
        num_samples = None,
        eval_every = 1000,
        results_folder = './results',
        mixed_precision = 'no',
        seed=43,
    ):
        super().__init__()


        set_seeds(seed)

        self.args = args

        self.best_val_metric = 0
        self.num_samples = num_samples

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(
            mixed_precision = mixed_precision,
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs]
        )
        self.num_devices = self.accelerator.num_processes
        args.num_devices = self.num_devices

        if self.accelerator.is_main_process:
            if args.output_dir is None:
                args.output_dir = file_utils.get_output_dir(args)
            results_folder = args.output_dir
            with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
            run = os.path.split(__file__)[-1].split(".")[0]
            if args.wandb_name:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder, "name": args.wandb_name}})
            else:
                self.accelerator.init_trackers(run, config=args, init_kwargs={"wandb": {"dir": results_folder}})

        self.enc_dec_model = args.enc_dec_model

        self.lm, config = get_latent_model(args)
        num_trainable_params = sum(p.numel() for p in self.lm.parameters() if p.requires_grad)
        if self.accelerator.is_main_process:
            self.accelerator.print(f'num trainable params: {num_trainable_params}')

        self.eval_every = eval_every

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.train_num_steps = train_num_steps

        # dataset and dataloader
        self.train_data, self.val_data = get_dataset(args)
        self.train_dataset = MelodyDataset(self.args, self.train_data)
        self.val_dataset = MelodyDataset(self.args, self.val_data)

        if args.eval:
            self.train_dataset = MelodyDataset(self.args, {key: value[:1000] for key, value in self.train_data.items()})
        self.dataloader = get_dataloader(args, self.train_dataset, config, shuffle=True)
        self.val_dataloader = get_dataloader(args, self.val_dataset, config, shuffle=True)
        self.max_seq_len = args.max_seq_len

        # optimizer

        # check that the new latents are added to the optimizer
        if args.optimizer == 'adamw':
            self.opt = optimizer.get_adamw_optimizer(self.lm.parameters(), lr = train_lr, betas = adam_betas, weight_decay=adam_weight_decay)
        elif args.optimizer == 'lion':
            self.opt = optimizer.get_lion_optimizer(self.lm.parameters(), lr = train_lr, weight_decay=adam_weight_decay)
        else:
            raise ValueError(f'invalid optimizer {args.optimizer}')

        # scheduler

        lr_scheduler = get_scheduler(
            lr_schedule,
            optimizer=self.opt,
            num_warmup_steps=num_warmup_steps*self.num_devices,
            num_training_steps=train_num_steps*self.num_devices,
        )

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.lm, self.opt, self.dataloader, self.lr_scheduler, self.val_dataloader = self.accelerator.prepare(self.lm, self.opt, self.dataloader, lr_scheduler, self.val_dataloader)
        self.data_iter = cycle(self.dataloader)
        self.val_iter = cycle(self.val_dataloader)

    def save(self):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.lm),
            'opt': self.opt.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model.pt'))

    def load(self, file_path=None, resume_training=False, only_bart=False):
        file_path = Path(file_path) if exists(file_path) else self.results_folder
        accelerator = self.accelerator
        device = accelerator.device

        model = self.accelerator.unwrap_model(self.lm)
        data = torch.load(str(file_path / f'bart_model.pt'), map_location=device, weights_only=True)        
        model.load_state_dict(data['model'], strict=False)
        if not only_bart:
            data = torch.load(str(file_path / f'model.pt'), map_location=device, weights_only=True)
            model.load_state_dict(data['model'])
            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            # Advance lr scheduler to current step
            if resume_training:
                for _ in range(self.step):
                    self.lr_scheduler.step()            
    
    def validation(self):
        self.lm.eval()
        pred_text = {k:[] for k,_ in generate_kwargs.items()}    
        bart_text = {k:[] for k,_ in generate_kwargs.items()}    
        ref_text = []
        accelerator = self.accelerator
        device = self.accelerator.device
        for batch in tqdm(self.val_dataloader):
            for strategy in generate_kwargs.keys():
                gen_kwargs = generate_kwargs[strategy]
                gen_kwargs['max_length'] = self.max_seq_len
                data = {k:v.to(device) for k,v in batch.items()}
                # Compute generated language
                texts_list = []
                for i in range(len(data['input_ids'][0])):
                    if self.num_devices > 1:
                        encoder_outputs = self.lm.module.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                        encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'][:,i])
                        sample_ids = self.lm.module.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                    else:
                        encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                        encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'][:,i])
                        sample_ids = self.lm.generate(encoder_outputs=encoder_outputs, **gen_kwargs)
                    # Pad sample_ids to max_seq_len
                    sample_ids = F.pad(sample_ids, (0, self.max_seq_len - sample_ids.shape[-1]), value=1)
                    gathered_sample_ids = accelerator.gather(sample_ids).to('cpu')
                    if texts_list == []:
                        texts_list = [''.join(chr(idx) if idx > 3 else '' for idx in g if idx != 3) for g in gathered_sample_ids]
                    else:
                        assert len(texts_list) == len(sample_ids)
                        for j in range(len(texts_list)):
                            texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > 3 else '' for idx in gathered_sample_ids[j] if idx != 3)}"
                texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                pred_text[strategy].extend(texts_list)

                # Compute BART language
                texts_list = []
                for i in range(len(data['input_ids'][0])):
                    if self.num_devices > 1:
                        sample_ids2 = self.lm.module.generate(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i], **gen_kwargs)
                    else:
                        sample_ids2 = self.lm.generate(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i], **gen_kwargs)
                    sample_ids2 = F.pad(sample_ids2, (0, self.max_seq_len - sample_ids2.shape[-1]), value=1)
                    gathered_sample_ids2 = accelerator.gather(sample_ids2).to('cpu')
                    if texts_list == []:
                        texts_list = [''.join(chr(idx) if idx > 3 else '' for idx in g if idx != 3) for g in gathered_sample_ids2]
                    else:
                        assert len(texts_list) == len(sample_ids)
                        for j in range(len(texts_list)):
                            texts_list[j] = f"{texts_list[j]}|{''.join(chr(idx) if idx > 3 else '' for idx in gathered_sample_ids2[j] if idx != 3)}"
                texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
                bart_text[strategy].extend(texts_list)

            # Store reference language
            gathered_input_ids = accelerator.gather(data['input_ids']).to('cpu')
            texts_list = []
            for datas in gathered_input_ids.numpy().tolist():
                ref_list = []
                for texts in datas:
                    ref_list.extend(texts)
                    ref_list.append(ord('|'))
                ref_list = ref_list[:-1]
                texts_list.append(''.join(chr(idx) if idx > 3 else ''  for idx in ref_list if idx != 3))
            texts_list = [text.strip() for text in texts_list if len(text.strip())>0]
            ref_text.extend(texts_list)
            if len(ref_text) > 1000:
                break

        if not self.accelerator.is_main_process:
            return
        # Compute metrics
        metrics = {}
        for strategy in generate_kwargs.keys():
            # Compute BLEU score
            metrics[f'autoencoder/{strategy}/bleu'] = evaluation.compute_bleu(pred_text[strategy], ref_text)
            metrics[f'bart/{strategy}/bleu'] = evaluation.compute_bleu(bart_text[strategy], ref_text)
            # Compute perplexity

            if all(pred_text[strategy]):
                metrics[f'autoencoder/{strategy}/perplexity'] = evaluation.compute_perplexity(pred_text[strategy])

            if all(bart_text[strategy]):
                metrics[f'bart/{strategy}/perplexity'] = evaluation.compute_perplexity(bart_text[strategy])

            rouge_metrics = evaluation.compute_rouge(pred_text[strategy], ref_text)
            for k,v in rouge_metrics.items():
                metrics[f'autoencoder/{strategy}/{k}'] = v
            rouge_metrics = evaluation.compute_rouge(bart_text[strategy], ref_text)
            for k,v in rouge_metrics.items():
                metrics[f'bart/{strategy}/{k}'] = v
        metrics['reference/perplexity'] = evaluation.compute_perplexity(ref_text)
         
        accelerator.log(metrics, self.step)
        print(metrics)

        # Log samples
        # reference | strategy0/autoencoder | strategy0/bart | strategy1/autoencoder | strategy1/bart | ...
        columns = ['reference'] + [f'{strategy}/autoencoder' for strategy in generate_kwargs.keys()] + [f'{strategy}/bart' for strategy in generate_kwargs.keys()]
        data = []
        for i in range(len(ref_text)):
            row = [ref_text[i]]
            for strategy in generate_kwargs.keys():
                row.append(pred_text[strategy][i])
            
            for strategy in generate_kwargs.keys():
                row.append(bart_text[strategy][i])
            data.append(row)
        table = wandb.Table(columns=columns, data=data)
        accelerator.log({f"Samples": table}, self.step)
        with open( f'{self.args.save_dir}/prediction.json', "w") as file:
            json.dump(data, file, ensure_ascii=False)
        print(f'prediction saved at ~/prediction.json')

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        self.lm.train()
        if self.args.lm_mode == 'freeze':
            encoder_context = torch.no_grad()
        else:
            encoder_context = nullcontext()

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            if self.step != 0:
                data = next(islice(self.data_iter, (self.step//self.args.seg)-1, None))
                if self.step >= 1000:
                    val_data = next(islice(self.val_iter, (self.step//1000)-1, None))

            while self.step < self.train_num_steps:

                total_loss = 0.

                data = {k:v.to(device) for k,v in next(self.data_iter).items()}

                with accelerator.autocast():
                    for i in range(len(data['input_ids'][0])):
                        with encoder_context:
                            if self.num_devices > 1:
                                encoder_outputs = self.lm.module.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                            else:
                                encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                        if self.num_devices > 1:
                            encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'][:,i])
                        else:
                            encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'][:,i])

                        loss = self.lm(labels=data['labels'][:,i].clone(), encoder_outputs=encoder_outputs).loss     
                        total_loss += loss.item()

                        self.accelerator.backward(loss)

                        accelerator.wait_for_everyone()

                        grad_norm = compute_grad_norm(self.lm.parameters())

                        accelerator.clip_grad_norm_(self.lm.parameters(), 1.0)
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()

                        accelerator.wait_for_everyone()

                        self.step += 1

                # Log to WandB
                if self.step % 1000 == 0:
                    self.lm.eval()
                    with torch.no_grad():
                        total_val_loss = 0.
                        total_lm_val_loss = 0.
                        data = {k:v.to(device) for k,v in next(self.val_iter).items()}
                        for i in range(len(data['input_ids'][0])):
                            if self.num_devices > 1:
                                encoder_outputs = self.lm.module.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                                encoder_outputs = self.lm.module.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'][:,i])
                            else:
                                encoder_outputs = self.lm.get_encoder()(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i])
                                encoder_outputs = self.lm.encoder_output_to_decoder_input(encoder_outputs, data['attention_mask'][:,i])
                            loss = self.lm(labels=data['labels'][:,i].clone(), encoder_outputs=encoder_outputs).loss                      
                            if self.args.lm_mode == 'freeze':
                                total_lm_val_loss += self.lm(input_ids = data['input_ids'][:,i], attention_mask = data['attention_mask'][:,i], labels=data['labels'][:,i].clone()).loss.item()
                            total_val_loss += loss.item()

                        logs = {"train/loss": total_loss / len(data['input_ids'][0]), "val/loss": total_val_loss / len(data['input_ids'][0]), "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0],
                                "step": self.step, "epoch": (self.step)/(len(self.dataloader)*len(data['input_ids'][0])), "samples": self.step*self.train_batch_size*self.num_devices}
                        if self.args.lm_mode == 'freeze':
                            logs["val/lm_loss"] = total_lm_val_loss / len(data['input_ids'][0])
                        pbar.set_postfix(**logs)                            
                    print("logs : ", logs)
                    self.lm.train()
                else:
                    logs = {"train/loss": total_loss / len(data['input_ids'][0]), "grad_norm": grad_norm, "lr": self.lr_scheduler.get_last_lr()[0], "step": self.step,
                            "epoch": (self.step)/(len(self.dataloader)*len(data['input_ids'][0])), "samples": self.step*self.train_batch_size*self.num_devices}

                if accelerator.is_main_process:
                    accelerator.log(logs, step=self.step)

                if self.step % self.eval_every == 0:
                    self.save()
                    self.validation()
                    accelerator.wait_for_everyone()                    
                    self.lm.train() 

                pbar.update(self.args.seg)
        self.validation()
        self.save()

        accelerator.print('training complete')