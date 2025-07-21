import sys
import os

import pretty_midi
import textgrid
import re
import numpy as np

import torch
import torch.nn as nn
from torch.optim import AdamW

from datetime import datetime
from pathlib import Path
from spacy.symbols import ORTH

base_tones = {
    'C' : 0, 'C#': 1, 'D' : 2, 'D#': 3,
    'E' : 4, 'F' : 5, 'F#': 6, 'G' : 7,
    
    'G#': 8, 'A' : 9, 'A#':10, 'B' :11,
}
abc_tones = {
    'C', 'D', 'E', 'F',    
    'G', 'A', 'B',
}
abc_tones_h = {
    'c', 'd', 'e', 'f',    
    'g', 'a', 'b',
}
line_index = {
    0: 'first', 1 : 'second', 2: 'third',
    3 : 'fourth', 4 : 'fifth', 
    5: 'sixth', 6 : 'seventh',
    7: 'eighth', 8 : 'ninth', 9: 'tenth',
}


def log_discretize(x, bins=1024):
    eps = 1
    x_min = np.log(eps-0.09)
    x_max = np.log(8.1+eps)
    x = min(8.1, x)
    x = max(-0.09, x)
    x = np.log(x+eps)
    x = (x-x_min) / (x_max-x_min) * (bins-1)
    return np.round(x).astype(int)

def reverse_log_float(x, bins=1024):
    if x == 42:
        return 0
    eps = 1
    x_min = np.log(eps-0.09)
    x_max = np.log(8.1+eps)
    x = x * (x_max - x_min)/(bins-1) + x_min
    x = np.exp(x) - eps
    return float("{:.3f}".format(x))

def bin_time(list_d):
    bin_list = []
    for item in list_d:
        if not isinstance(item, str):
            item = str(item)
        item_tuple = item.split(' ')
        out = ''
        for item_str in item_tuple:
            item_num = float(item_str)
            # out += f'<{item_num}>'
            bin = log_discretize(item_num)
            out += f'<{bin}>'
        bin_list.append(out)
    return bin_list

def append_song_token(args, model, tokenizer):
    old_token_len = len(tokenizer)
    new_tokens = ['<bop>','<eop>','<eos>','<rest>','<sos>']
    for note in base_tones:
        for i in range(-1, 10): # -1 -> 9
            new_tokens.append(f'<{note}{i}>') 
    for t_bin in range(1024):
        new_tokens.append(f'<{t_bin}>')
    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
    new_tokens = list(new_tokens)
    new_tokens.sort()
    tokenizer.add_tokens(new_tokens)
    new_token_len = len(tokenizer)
    model.tokenizer = tokenizer    

    if 'bart' in args.enc_dec_model:
        model.config.vocab_size = new_token_len
        model.resize_token_embeddings(new_token_len)
    elif 'bert' in args.enc_dec_model:
        model.encoder.config.vocab_size = new_token_len
        model.decoder.config.vocab_size = new_token_len
        model.encoder.resize_token_embeddings(new_token_len)
        model.decoder.resize_token_embeddings(new_token_len)
    else:
        model.config.vocab_size = new_token_len
        model.resize_token_embeddings(new_token_len)

    return model, tokenizer

def append_song_token_abc(args, model, tokenizer):
    new_tokens = ['<bop>','<eop>','<eos>','<rest>','<sos>']
    for note in abc_tones:
        new_tokens.append(f'{note},')
        new_tokens.append(f'{note},,')
    for note in abc_tones_h:
        new_tokens.append(f"{note}'")
    new_tokens = set(new_tokens) - set(tokenizer.get_vocab().keys())
    new_tokens = list(new_tokens)
    new_tokens.sort()
    tokenizer.add_tokens(new_tokens)
    new_token_len = len(tokenizer)
    model.tokenizer = tokenizer    

    if 'bart' in args.enc_dec_model:
        model.config.vocab_size = new_token_len
        model.resize_token_embeddings(new_token_len)
    elif 'bert' in args.enc_dec_model:
        model.encoder.config.vocab_size = new_token_len
        model.decoder.config.vocab_size = new_token_len
        model.encoder.resize_token_embeddings(new_token_len)
        model.decoder.resize_token_embeddings(new_token_len)
    else:
        model.config.vocab_size = new_token_len
        model.resize_token_embeddings(new_token_len)

    return model, tokenizer

def append_song_token_abc_tokenizer(tokenizer):

    new_tokens = ['<bop>','<eop>','<eos>','<rest>','<sos>']
    for note in abc_tones:
        new_tokens.append(f'{note},')
        new_tokens.append(f'{note},,')
    for note in abc_tones_h:
        new_tokens.append(f"{note}'")
    new_tokens = set(new_tokens) - set(tokenizer.vocab)
    new_tokens = list(new_tokens)
    new_tokens.sort()
    for token in new_tokens:
        tokenizer.add_special_case(token, [{ORTH:  token}])

    return tokenizer

def append_song_token_tokenizer(tokenizer):

    new_tokens = ['<bop>','<eop>','<eos>','<rest>','<sos>']
    for note in base_tones:
        for i in range(-1, 10): # -1 -> 9
            new_tokens.append(f'<{note}{i}>') 
    for t_bin in range(1024):
        new_tokens.append(f'<{t_bin}>')
    new_tokens = set(new_tokens) - set(tokenizer.vocab)
    new_tokens = list(new_tokens)
    new_tokens.sort()
    for token in new_tokens:
        tokenizer.add_special_case(token, [{ORTH:  token}])

    return tokenizer

def tuple2dict(line):
    order_string = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
    line = line.replace(" ", "")
    line = line.replace("\n", "")
    line = line.replace("<bop>", "")
    line = line.replace("<eos>", "")
    line = line.replace("<eop>", "")
    line = line.replace("<sos>", "")
    line = re.sub(r'\. |\.', '', line)
    # line = re.sub(r'The\d+line:', ' |', line)
    for string in order_string:
        line = line.replace(f'The{string}line:', ' |')
    special_pattern = r'<(.*?)>'
    song = {'lyrics':[], 'notes':[], 'notes_duration':[], 'rest_duration':[], 'pitch':[], 'notes_dict': [], 'rest_dict': []}
     
    for item in line.split('|'):
        x = item.split(',')
        notes = re.findall(special_pattern,x[1])
        note_ds = re.findall(special_pattern,x[2])
        rest_d = re.findall(special_pattern,x[3])[0]
        assert len(notes)== len(note_ds), f"notes:{'|'.join(notes)}, note_ds:{'|'.join(note_ds)}"
        for i in range(len(notes)):
            if i == 0:
                song['lyrics'].append(x[0])
            else:
                song['lyrics'].append('-')
            song['notes'].append(notes[i])
            #song['pitch'].append(int(pretty_midi.note_name_to_number(notes[i])))
            song['notes_duration'].append(reverse_log_float(int(note_ds[i])))
            #song['notes_dict'].append(int(note_ds[i]))
            if i == len(notes)-1:
                song['rest_duration'].append(reverse_log_float(int(rest_d)))
                #song['rest_dict'].append(int(rest_d))
            else:
                song['rest_duration'].append(0)
                #song['rest_dict'].append(0)
    return song

def dict2midi(song, match = None):
    # new_midi = pretty_midi.PrettyMIDI(charset="utf-8")#
    new_midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    # print(len(song["notes"]))
    current_time = 0  # Time since the beginning of the song, in seconds
    #pitch = []
    lyrics = ''
    i, add_number = 0, 0
    ph_word_match = []

    while i < len(song["notes"]):
        #add notes
        notes_duration = 0.000
        rest_duration = 0.000
        notes_duration += song["notes_duration"][i]
        rest_duration += song["rest_duration"][i]
        step = 0
        if song["notes"][i] == 'rest':
            add_number += 1
            step += 1
            pass
        else:
            if match:
                if song["lyrics"][i] != '-':
                    if i < (len(song["notes"])-1) and (i+1-add_number) < len(match) and song["lyrics"][i+1] != '-' \
                        and match[i-add_number] == match[i+1-add_number] and song["notes"][i] == song["notes"][i+1]:
                        lyrics += f'{song["lyrics"][i]}{song["lyrics"][i+1]} '
                        ph_word_match.append(match[i-add_number])
                        notes_duration += song["notes_duration"][i+1]
                        rest_duration += song["rest_duration"][i+1]
                        step += 2
                    else:
                        lyrics += f'{song["lyrics"][i]} '
                        ph_word_match.append(match[i-add_number])
                        step += 1
                else:
                    add_number += 1
                    lyrics += f'{song["lyrics"][i]} '
                    ph_word_match.append('-')
                    step += 1
            else:
                lyrics += f'{song["lyrics"][i]} '
                step += 1

            note_obj = pretty_midi.Note(velocity=100, pitch=int(pretty_midi.note_name_to_number(song["notes"][i])), start=current_time,
                                    end=current_time + notes_duration)
            instrument.notes.append(note_obj)
            #add lyrics
            # lyrics += f'{song["lyrics"][i]} '
            # lyric_event = pretty_midi.Lyric(text=str(song["lyrics"][i])+ "\0", time=current_time)
            # new_midi.lyrics.append(lyric_event)
        current_time +=  notes_duration + rest_duration# Update of the time
        i += step
   
    new_midi.instruments.append(instrument)
    #lyrics = ' '.join(song["lyrics"])
    if len(lyrics) > 0:
        lyrics = lyrics[:-1]
    return new_midi, lyrics, ph_word_match

def gen_midi(line, file_name, match_file = None):
    match = None
    if match_file:
        with open(match_file, 'r') as f:
            match = f.read()
        match = list(match.split(","))
    song  = tuple2dict(line)
    #song['lyrics'] = ['I','-','you','-','I','-','you','-','I','-','you','-','he','-']
    new_midi, lyrics, ph_word_match = dict2midi(song, match)    
    
    # save midi file and lyric text
    new_midi.write(file_name+'.mid')
    
    with open(file_name+'.txt', "w") as file:
        file.write(lyrics)

    if match_file:
        ph_word_match2string = ','.join(ph_word_match)
        with open(file_name+'_ph_word_match.txt', "w") as file:
            file.write(ph_word_match2string)

    print(f'midi saved at ~/{file_name}.mid, lyrics saved at ~/{file_name}.txt, ph_word_match saved at ~/{file_name}_ph_word_match.txt')

def tuple2dict_fix(lines):
    songs = []
    for i in range(len(lines)):
        line = lines[i]
        order_string = ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth']
        line = line.replace(" ", "")
        line = line.replace("\n", "")
        line = line.replace("<bop>", "")
        line = line.replace("<eos>", "")
        line = line.replace("<eop>", "")
        line = line.replace("<sos>", "")
        line = re.sub(r'\. |\.', '', line)
        # line = re.sub(r'The\d+line:', ' |', line)
        for string in order_string:
            line = line.replace(f'The{string}line:', ' |')
        lines[i] = line

    for i in range(len(lines)):
        special_pattern = r'<(.*?)>'
        song = {'lyrics':[], 'notes':[], 'notes_duration':[], 'rest_duration':[], 'pitch':[], 'notes_dict': [], 'rest_dict': []}
        line = lines[i]
        for j in range(len(line.split('|'))):
            x = line.split('|')[j].split(',')
            try:
                notes = re.findall(special_pattern,x[1])
                note_ds = re.findall(special_pattern,x[2])
                rest_d = re.findall(special_pattern,x[3])[0]            
                assert len(notes) == len(note_ds), f"notes:{'|'.join(notes)}, note_ds:{'|'.join(note_ds)}"
                for k in range(len(notes)):
                    if (x[0].lower() == 'sp' or x[0].lower() == 'ap') and notes[k] == 'rest':
                        pass
                    else:
                        pretty_midi.note_name_to_number(notes[k])
                    reverse_log_float(int(note_ds[k]))
                reverse_log_float(int(rest_d))
            except:
                count = 0
                for l in range(len(lines)):
                    if l != i:
                        x = lines[l].split('|')[j].split(',')
                        print(f'i : {i}, l : {l}, j : {j}, x : {x}')
                        try:
                            notes = re.findall(special_pattern,x[1])
                            note_ds = re.findall(special_pattern,x[2])
                            rest_d = re.findall(special_pattern,x[3])[0]                        
                            assert len(notes)== len(note_ds), f"notes:{'|'.join(notes)}, note_ds:{'|'.join(note_ds)}"
                            for k in range(len(notes)):
                                if (x[0].lower() == 'sp' or x[0].lower() == 'ap') and notes[k] == 'rest':
                                    pass
                                else:
                                    pretty_midi.note_name_to_number(notes[k])
                                reverse_log_float(int(note_ds[k]))                                
                            reverse_log_float(int(rest_d))                            
                            break
                        except:
                            count += 1
                            if count == (len(lines) - 1):
                                print('All failed')
                                return
                            else:
                                pass
            
            for k in range(len(notes)):
                if k == 0:
                    song['lyrics'].append(x[0])
                else:
                    song['lyrics'].append('-')
                song['notes'].append(notes[k])
                #song['pitch'].append(int(pretty_midi.note_name_to_number(notes[k])))
                song['notes_duration'].append(reverse_log_float(int(note_ds[k])))
                #song['notes_dict'].append(int(note_ds[k]))
                if k == len(notes)-1:
                    song['rest_duration'].append(reverse_log_float(int(rest_d)))
                    #song['rest_dict'].append(int(rest_d))
                else:
                    song['rest_duration'].append(0)
                    #song['rest_dict'].append(0)

        songs.append(song)

    return songs

def dict2midi_fix(songs, match = None):
    # new_midi = pretty_midi.PrettyMIDI(charset="utf-8")#    
    #pitch = []
    
    new_midis = []
    lyrics_list = []
    ph_word_match_list = []    

    for song in songs:
        new_midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)
        current_time = 0  # Time since the beginning of the song, in seconds
        lyrics = ''
        i, add_number = 0, 0
        ph_word_match = []
        while i < len(song["notes"]):
            #add notes
            notes_duration = 0.000
            rest_duration = 0.000
            notes_duration += song["notes_duration"][i]
            rest_duration += song["rest_duration"][i]
            step = 0
            if song["notes"][i] == 'rest':
                add_number += 1
                step += 1
                pass
            else:
                if match:
                    if song["lyrics"][i] != '-':
                        if i < (len(song["notes"])-1) and (i+1-add_number) < len(match) and song["lyrics"][i+1] != '-' \
                            and match[i-add_number] == match[i+1-add_number] and song["notes"][i] == song["notes"][i+1]:
                            lyrics += f'{song["lyrics"][i]}{song["lyrics"][i+1]} '
                            ph_word_match.append(match[i-add_number])
                            notes_duration += song["notes_duration"][i+1]
                            rest_duration += song["rest_duration"][i+1]
                            step += 2
                        else:
                            lyrics += f'{song["lyrics"][i]} '
                            ph_word_match.append(match[i-add_number])
                            step += 1
                    else:
                        add_number += 1
                        lyrics += f'{song["lyrics"][i]} '
                        ph_word_match.append('-')
                        step += 1
                else:
                    lyrics += f'{song["lyrics"][i]} '
                    step += 1

                note_obj = pretty_midi.Note(velocity=100, pitch=int(pretty_midi.note_name_to_number(song["notes"][i])), start=current_time,
                                            end=current_time + notes_duration)
                instrument.notes.append(note_obj)
                #add lyrics
                # lyrics += f'{song["lyrics"][i]} '
                # lyric_event = pretty_midi.Lyric(text=str(song["lyrics"][i])+ "\0", time=current_time)
                # new_midi.lyrics.append(lyric_event)
            current_time +=  notes_duration + rest_duration# Update of the time
            i += step
    
        new_midi.instruments.append(instrument)
        #lyrics = ' '.join(song["lyrics"])
        if len(lyrics) > 0:
            lyrics = lyrics[:-1]

        new_midis.append(new_midi)
        lyrics_list.append(lyrics)
        ph_word_match_list.append(ph_word_match)

    return new_midis, lyrics_list, ph_word_match_list

def gen_midi_fix(lines, file_name, match_file = None):
    match = None
    if match_file:
        with open(match_file, 'r') as f:
            match = f.read()
        match = list(match.split(","))
    songs  = tuple2dict_fix(lines)
    #song['lyrics'] = ['I','-','you','-','I','-','you','-','I','-','you','-','he','-']
    new_midis, lyrics_list, ph_word_match_list = dict2midi_fix(songs, match)    
    
    # save midi file and lyric text
    for i in range(len(new_midis)):
        new_midis[i].write(f'{i+42}_{file_name}.mid')
    
        with open(f'{i+42}_{file_name}.txt', "w") as file:
            file.write(lyrics_list[i])

        if match_file:
            ph_word_match2string = ','.join(ph_word_match_list[i])
            with open(f'{i+42}_{file_name}_ph_word_match.txt', "w") as file:
                file.write(ph_word_match2string)

        print(f'midi saved at ~/{i+42}_{file_name}.mid, lyrics saved at ~/{i+42}_{file_name}.txt, ph_word_match saved at ~/{i+42}_{file_name}_ph_word_match.txt')

def pinyin_to_word(tg_file, lyrics_file):
    tg = textgrid.TextGrid.fromFile(tg_file)
    pin_tg = tg[2]

    with open(lyrics_file, 'r', encoding='UTF-8') as f:
        lyrics = f.read()
    lyrics = list(lyrics.split(" "))
    word_lyrics = ''
    i, j = 0, 0
    while i < len(lyrics) and j < len(pin_tg):
        if lyrics[i] != '-':
            if lyrics[i] == pin_tg[j].mark:
                word_lyrics += tg[1][j].mark
                i += 1
                j += 1
            elif pin_tg[j].mark == '_' or pin_tg[j].mark.lower() == 'sp' or pin_tg[j].mark.lower() == 'ap':
                j += 1
            else:
                i += 1
        else :
            word_lyrics += '#'
            i += 1

    lyrics = ' '.join(lyrics)
    with open(lyrics_file, "w", encoding='UTF-8') as file:
        file.write(lyrics + '\n')
        file.write('\n')
        file.write(word_lyrics)

def phonem_to_word(tg_file, lyrics_file, ph_word_match_file):
    tg = textgrid.TextGrid.fromFile(tg_file)

    with open(lyrics_file, 'r', encoding='UTF-8') as f:
        lyrics = f.read()

    with open(ph_word_match_file, 'r') as f:
        ph_word_match = f.read()
    ph_word_match = list(ph_word_match.split(","))

    word_lyrics = ''
    i = 0
    while i < len(ph_word_match):
        number = ph_word_match[i]
        if number != '-':
            word_lyrics += tg[1][int(number)].mark
            i += 1
            if i < len(ph_word_match):
                next_number = ph_word_match[i]
                if next_number != '-' and number == next_number:
                    word_lyrics += '#'
                    i += 1
        else:
            word_lyrics += '#'
            i += 1

    with open(lyrics_file, "w", encoding='UTF-8') as file:
        file.write(lyrics + '\n')
        file.write('\n')
        file.write(word_lyrics)

def separate_weight_decayable_params(params):
    # Exclude affine params in norms (e.g. LayerNorm, GroupNorm, etc.) and bias terms
    no_wd_params = [param for param in params if param.ndim < 2]
    wd_params = [param for param in params if param not in set(no_wd_params)]
    return wd_params, no_wd_params

def get_adamw_optimizer(params, lr, betas, weight_decay, eps=1e-8):
    params = list(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = weight_decay, betas=betas, eps=eps)

def compute_grad_norm(parameters):
    # implementation adapted from https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), p=2) for p in parameters]), p=2).item()
    return total_norm

def get_output_dir():
    output_dir = f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    os.makedirs(output_dir, exist_ok = True)
    print(f'Created {output_dir}')
    return output_dir