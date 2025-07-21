import json
import random
import argparse
import re
from unidecode import unidecode

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartConfig, BartForConditionalGeneration, default_data_collator
from datasets import load_dataset

from dataset_utils.denoising_collator import DataCollatorForBartDenoisingLM
from dataset_utils.flan_collator import DataCollatorForFlanLM


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def load_json(json_path):
    json_path = json_path.replace('\\', '/')
    json_file = open(json_path, 'r', encoding='utf-8')
    return json.load(json_file)

def split_data(data, eval_ratio=0.1):
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set

def get_dataset(args):
    if args.dataset_name == 'irishman':
        dataset = load_dataset('sander-wood/irishman')
        train_data, val_data = process_irishman_dataset(args, dataset)
    return train_data, val_data

def clean_data(text):
    text = unidecode(text)
    patterns = [r'"(.*?)"', r'!(.*?)!']
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    # text = text.replace(':', '')
    text = text.replace(': ', ':')
    text = text.replace(' :', ':')
    delimiters = ["::", "[|", "||", "|]"]
    regexPattern = '|'.join(map(re.escape, delimiters))
    text = re.split(regexPattern, text)
    text = list(filter(None, text))

    return text

def segment_data(args, text, context=None):
    segment_text = []
    segment_context = []
    if args.seq2seq:
        context = unidecode(context)
    patterns = [r"\[M:.+\]", r"\[K:.+\]", r"\[Q:.+\]"]
    for pattern in patterns:
        matches = re.search(pattern, text)
        if bool(matches):
            break
    if bool(matches):
        pass
    else:
        cleaned_text = clean_data(text)
        for clean_text in cleaned_text:
            lines = clean_text.split('|')
            if args.strip:
                lines = [line.strip() for line in lines]
            lines = list(filter(None, lines))
            if args.min != None:
                lines = list(filter(lambda x: len(x) >= args.min, lines))
            if args.strip:
                lines = [line.strip() for line in lines]
            if len(lines) < args.seg:
                # segment_text.append(line)
                # if args.seq2seq:
                #     segment_context.append(context)
                pass
            else:
                for i in range(0, len(lines), args.seg):
                    skip = False
                    if i >= len(lines) - args.seg + 1:
                        for line in lines[-args.seg:]:
                            if len(line) >= 32:
                                skip = True
                                break
                        if not skip:
                            segment_text.append(lines[-args.seg:])
                            if args.seq2seq:
                                segment_context.append(context)
                    else:
                        for line in lines[i:i+args.seg]:
                            if len(line) >= 32:
                                skip = True
                                break
                        if not skip:
                            segment_text.append(lines[i:i+args.seg])
                            if args.seq2seq:
                                segment_context.append(context)
    return segment_text, segment_context

def process_irishman_dataset(args, dataset):
    def extract_irishman_text(args, example, new_data):
        lines = example.split('\n')
        text = None
        context = None
        note_length = lines[1].split(':')[-1]
        if args.seq2seq:
            if args.L != None and args.L != note_length:
                pass
            elif lines[2][0] == 'M':
                meter = lines[2].split(':')[-1]
                if args.M != None and args.M != meter:
                    pass
                else:
                    context = " ".join(lines[1:4])
                    text = "".join(lines[4:])
            elif lines[3][0] == 'M':
                meter = lines[3].split(':')[-1]
                if args.M != None and args.M != meter:
                    pass
                else:
                    context = " ".join(lines[1:5])
                    text = "".join(lines[5:])
            else:
                pass
        else:
            if args.L != None and args.L != note_length:
                pass
            elif lines[2][0] == 'M':
                meter = lines[2].split(':')[-1]
                if args.M != None and args.M != meter:
                    pass
                else:
                    text = "".join(lines[4:])
            elif lines[3][0] == 'M':
                meter = lines[3].split(':')[-1]
                if args.M != None and args.M != meter:
                    pass
                else:
                    text = "".join(lines[5:])
            else:
                pass
        if text != None:
            segment_text, segment_context = segment_data(args, text, context=context)
            if args.seq2seq:
                new_data['text'].extend(segment_text)
                new_data['context'].extend(segment_context)
            else:
                new_data['text'].extend(segment_text)

        return new_data

    train_data = dataset['train']['abc notation']
    valid_data = dataset['validation']['abc notation']

    if args.seq2seq:
        new_train_data, new_val_data = {'text' : [], 'context' : []}, {'text' : [], 'context' : []}
    else:
        new_train_data, new_val_data = {'text' : []}, {'text' : []}

    for example in train_data:
        new_train_data = extract_irishman_text(args, example, new_train_data)
    for example in valid_data:
        new_val_data = extract_irishman_text(args, example, new_val_data)

    return new_train_data, new_val_data

# Define the custom dataset
class MelodyDataset(Dataset):
    def __init__(self, args, data):
        self.args = args
        self.data = data
        if 'flan-t5' in args.enc_dec_model:
            self.bos_token_id = None
            self.pad_token_id = 0
            self.eos_token_id = 1
        elif 'bert' in args.enc_dec_model:
            self.bos_token_id = 2
            self.pad_token_id = 0
            self.eos_token_id = 1
        else:
            self.bos_token_id = 0
            self.pad_token_id = 1
            self.eos_token_id = 2
        self.mask_token_id = 3

    def __len__(self):
        return len(self.data['text'])

    def __getitem__(self, idx):
        texts = self.data['text'][idx]
        if self.args.seq2seq:
            data = {'input_ids' : [], 'attention_mask' : [], 'labels' : [], 'cond_input_ids' : [], 'cond_attention_mask' : []}
        else:
            data = {'input_ids' : [], 'attention_mask' : [], 'labels' : []}
        for text in texts:
            text = unidecode(text)
            if 'flan-t5' in self.args.enc_dec_model:
                input_ids = [ord(c) for c in text] + [self.eos_token_id]
            else:
                input_ids = [self.bos_token_id] + [ord(c) for c in text] + [self.eos_token_id]
            input_ids = input_ids[:self.args.max_seq_len]
            input_ids += [self.pad_token_id] * (self.args.max_seq_len - len(input_ids))
            attention_mask = [1 if c != self.pad_token_id else 0 for c in input_ids]
            data["input_ids"].append(input_ids)
            data["attention_mask"].append(attention_mask)
            data["labels"].append(input_ids)
        if self.args.seq2seq:
            context_text = self.data['context'][idx]
            context_text = unidecode(context_text)
            if 'flan-t5' in self.args.enc_dec_model:
                cond_input_ids = [ord(c) for c in context_text] + [self.eos_token_id]
            else:
                cond_input_ids = [self.bos_token_id] + [ord(c) for c in context_text] + [self.eos_token_id]
            cond_input_ids = cond_input_ids[:self.args.context_max_seq_len]
            cond_input_ids += [self.pad_token_id] * (self.args.context_max_seq_len - len(cond_input_ids))
            cond_attention_mask = [1 if c != self.pad_token_id else 0 for c in cond_input_ids]
            data["cond_input_ids"].append(cond_input_ids)
            data["cond_attention_mask"].append(cond_attention_mask)

        return data

def get_dataloader(args, dataset, model_config, shuffle=False):
        # Create the dataloader
    if 'mbart' in args.enc_dec_model:
        collate_fn=default_data_collator
    elif 'bart' in args.enc_dec_model:
        collate_fn=DataCollatorForBartDenoisingLM(model_config.decoder_start_token_id)
    elif 't5' in args.enc_dec_model:
        collate_fn=DataCollatorForFlanLM()
    elif 'bert' in args.enc_dec_model:
        collate_fn=default_data_collator
    else:
        raise NotImplementedError

    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=args.train_batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 2
        )

    return dl

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default="irishman")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq2seq", action="store_true", default=False)

    # Optimization hyperparameters
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--context_max_seq_len", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--seg", type=int, default=4)
    parser.add_argument("--strip", action="store_true", default=False)
    parser.add_argument("--min", type=int, default=None)
    parser.add_argument("--L", type=str, default=None)
    parser.add_argument("--M", type=str, default=None)

    # Model hyperparemeters
    parser.add_argument("--enc_dec_model", type=str, default="facebook/bart-base")
    parser.add_argument("--hidden_size", type=int, default=768)

    # Load and eval model
    parser.add_argument("--eval", action="store_true", default=False)
    parser.add_argument("--eval_test", action="store_true", default=False)

    #args = parser.parse_args()
    args = parser.parse_args('--min 3 --L 1/8 --M 4/4 --strip --seg 8'.split())
    print(f'args : {args}')

    train_data, val_data = get_dataset(args)
    print(f'train_data length : {len(train_data)}, val_data length : {len(val_data)}')
    config = BartConfig.from_pretrained(args.enc_dec_model)
    config.max_position_embeddings = max(args.max_seq_len, 1024)
    config.vocab_size=args.vocab_size

    # Initialize the EncoderDecoderModel
    model = BartForConditionalGeneration(config = config)

    train_dataset = MelodyDataset(args, train_data)
    train_dataloader = get_dataloader(args, train_dataset, model.config, shuffle=False)
    # for batch in val_dataloader:
    #     for i in range(batch.input_ids.size()[0]):
    #         text = tokenizer.decode(batch.input_ids[i], skip_special_tokens=True)
    #         print(f'text : {text}')