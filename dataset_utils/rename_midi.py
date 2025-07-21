import os
import json
import glob
import shutil

def load_jsonl(jsonl_path):
    jsonl_path = jsonl_path.replace('\\', '/')
    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def print_jsonl(jsonl_path):

    data = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print(data[0])

def rename_midi(jsonl_path, midi_path, output_path):

    data = load_jsonl(jsonl_path)

    os.makedirs(output_path, exist_ok = True)

    for idx, line in enumerate(data):
        old_path = os.path.join(midi_path, f'{idx}_0.mid')
        print(f'old_path: {old_path}')
        file_name = line['file'].split('/')[1]
        new_path = os.path.join(output_path, file_name)
        print(f'new_path: {new_path}')

        shutil.copy(old_path, new_path)



if __name__ == "__main__":

    jsonl_path = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/10000x2/midi.jsonl"
    midi_path = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/10000x2/results/mid"
    output_path = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/10000x2/midi"
    
    rename_midi(jsonl_path, midi_path, output_path)