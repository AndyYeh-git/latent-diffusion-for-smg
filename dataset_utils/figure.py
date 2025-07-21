import os
import glob
import pretty_midi
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_piano_roll(midi_dir, output_dir, prompt=True):

    os.makedirs(output_dir, exist_ok = True)
    
    # Load the MIDI file
    midi_files = glob.glob(os.path.join(midi_dir, '*.mid'))
    for midi_file in midi_files:
        midi_file = midi_file.replace('\\', '/')
        file_name = midi_file.split('/')[-1][:-4]

        midi_data = pretty_midi.PrettyMIDI(midi_file)
        # Generate piano roll
        piano_roll = midi_data.get_piano_roll(fs=100)

        # 設定時間區段
        start_idx = 400
        end_idx = piano_roll.shape[1] - 400

        # Plot and save
        plt.figure(figsize=(12, 6))
        plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='hot', interpolation='nearest')
        plt.xlabel('Time (frames)')
        plt.ylabel('MIDI Note')
        plt.title(f'{file_name}.mid Piano Roll')

        if not prompt:
            # 加上框框 (x_start, y_start), width, height
            rect = patches.Rectangle((start_idx, 0), end_idx-start_idx, 127, linewidth=2, edgecolor='red', facecolor='none', label='Focus')
            plt.gca().add_patch(rect)

        plt.tight_layout()

        # Save to file
        
        plt_file = os.path.join(output_dir, f'{file_name}.png')
        plt_file = plt_file.replace('\\', '/')        
        plt.savefig(plt_file, dpi=300)

if __name__ == "__main__":

    midi_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/real_data/midi/prompt"    
    output_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/real_data/figures/prompt"
    prompt = False
    
    plot_piano_roll(midi_dir, output_dir, prompt)