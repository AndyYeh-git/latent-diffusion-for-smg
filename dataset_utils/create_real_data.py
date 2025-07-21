import os
import json
import glob
import numpy as np
import torch

import argparse
import re
from unidecode import unidecode
from datasets import load_dataset

def get_dataset(args):
    if args.dataset_name == 'irishman':
        dataset = load_dataset('sander-wood/irishman')
        process_irishman_dataset(args, dataset)
    else:
        raise NotImplementedError

def clean_data(args, text):
    text = unidecode(text)
    patterns = [r'"(.*?)"', r'!(.*?)!']
    for pattern in patterns:
        text = re.sub(pattern, '', text)
    # text = text.replace(':', '')
    text = text.replace(': ', ':')
    text = text.replace(' :', ':')
    if args.new:
        text = text.replace('[|', '|')
        text = text.replace('|]', '|')
        text = text.replace('||', '|')
        text = text.split('::')
    elif args.newest:
        delimiters = ["::", "[|", "||", "|]"]
        regexPattern = '|'.join(map(re.escape, delimiters))
        text = re.split(regexPattern, text)
    text = list(filter(None, text))

    return text

def segment_data(args, count, text, context=None, split='train'):
    patterns = [r"\[M:.+\]", r"\[K:.+\]", r"\[Q:.+\]"]
    for pattern in patterns:
        matches = re.search(pattern, text)
        if bool(matches):
            break
    if bool(matches):
        pass
    else:
        cleaned_text = clean_data(args, text)
        for clean_text in cleaned_text:
            lines = clean_text.split('|')
            if args.strip:
                lines = [line.strip() for line in lines]
            lines = list(filter(None, lines))
            if args.min != None:
                lines = list(filter(lambda x: len(x) >= args.min, lines))
                if args.strip:
                    lines = [line.strip() for line in lines]
                lines = list(filter(None, lines))

            if len(lines) < args.seg:
                pass
            else:
                # if not args.seg4bars:
                #     lines = [line[:31] for line in lines]
                for i in range(0, len(lines), args.seg):
                    skip = False
                    abc_dir = os.path.join(args.save_dir, split)
                    abc_file_path = os.path.join(abc_dir, f'{count}.abc')
                    abc_file_path = abc_file_path.replace('\\', '/')
                    if i >= len(lines) - args.seg + 1:
                        abc_data = []
                        abc_data.extend(context)
                        if args.seg4bars:
                            line = f'|{"|".join(lines[-4:])}|'
                            if len(line) > 128 or len(line) < 10:
                                continue
                            else:
                                abc_data.append(line)
                        else:
                            for line in lines[-args.seg:]:
                                if len(line) >= 32:
                                    skip = True
                                    break
                            if not skip:
                                abc_data.append("|".join(lines[-args.seg:]))
                        if not skip:
                            with open(abc_file_path, 'w') as abc_file:
                                for line in abc_data:
                                    abc_file.write(line + '\n')
                                if count % 2000 == 0:
                                    print(f'ABC notation saved to {abc_file_path}')
                    else:
                        abc_data = []
                        abc_data.extend(context)
                        if args.seg4bars:
                            line = f'|{"|".join(lines[i:i+4])}|'
                            if len(line) > 128 or len(line) < 10:
                                continue
                            else:
                                abc_data.append(line)
                        else:
                            for line in lines[i:i+args.seg]:
                                if len(line) >= 32:
                                    skip = True
                                    break
                            if not skip:
                                abc_data.append("|".join(lines[i:i+args.seg]))
                        if not skip:
                            with open(abc_file_path, 'w') as abc_file:
                                for line in abc_data:
                                    abc_file.write(line + '\n')
                                if count % 2000 == 0:
                                    print(f'ABC notation saved to {abc_file_path}')
                    if not skip:                    
                        count += 1
    return count

def process_irishman_dataset(args, dataset):
    def extract_irishman_text(args, example, count, split='train'):
        lines = example.split('\n')
        text = None
        context = None
        note_length = lines[1].split(':')[-1]
        if args.L != None and args.L != note_length:
            pass
        elif lines[2][0] == 'M':
            meter = lines[2].split(':')[-1]
            if args.M != None and args.M != meter:
                pass
            else:
                context = lines[1:4]
                text = "".join(lines[4:])
        elif lines[3][0] == 'M':
            meter = lines[3].split(':')[-1]
            if args.M != None and args.M != meter:
                pass
            else:
                context = lines[1:5]
                text = "".join(lines[5:])
        else:
            pass
        if text != None:
            count = segment_data(args, count, text, context, split=split)

        return count

    train_data = dataset['train']['abc notation']
    valid_data = dataset['validation']['abc notation']

    train_count, val_count = 0, 0

    os.makedirs(os.path.join(args.save_dir, 'train').replace('\\', '/'), exist_ok = True)
    os.makedirs(os.path.join(args.save_dir, 'val').replace('\\', '/'), exist_ok = True)

    for example in train_data:
        train_count = extract_irishman_text(args, example, train_count, split='train')
    for example in valid_data:
        val_count = extract_irishman_text(args, example, val_count, split='val')

    print(f'train_count : {train_count}')
    print(f'val_count : {val_count}')
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--dataset_name", type=str, default="irishman")
    parser.add_argument("--save_dir", type=str, default="../irishman_abc/L1_8_M4_4_seg_8_strip_min_3_final")

    # Optimization hyperparameters
    parser.add_argument("--seg", type=int, default=4)
    parser.add_argument("--strip", action="store_true", default=False)
    parser.add_argument("--min", type=int, default=None)
    parser.add_argument("--L", type=str, default=None)
    parser.add_argument("--M", type=str, default=None)
    parser.add_argument("--new", action="store_true", default=False)
    parser.add_argument("--newest", action="store_true", default=False)
    parser.add_argument("--seg4bars", action="store_true", default=False)

    #args = parser.parse_args()
    #args = parser.parse_args('--L 1/8 --M 4/4 --strip --new --seg4bars'.split())
    #args = parser.parse_args('--L 1/8 --M 4/4 --strip --newest --seg4bars'.split())
    #args = parser.parse_args('--L 1/8 --M 4/4 --strip --newest'.split())
    args = parser.parse_args('--min 3 --L 1/8 --M 4/4 --strip --newest --seg 8'.split())
    
    print(f'args : {args}')

    get_dataset(args)