import os
import glob
from midi2audio import FluidSynth

def midi_2_audio(SoundFont_path, midi_dir, output_dir):

    os.makedirs(output_dir, exist_ok = True)
    midi_files = glob.glob(os.path.join(midi_dir, '*.mid'))
    for midi_file in midi_files:
        midi_file = midi_file.replace('\\', '/')
        file_name = midi_file.split('/')[-1][:-4]

        # Initialize the FluidSynth synthesizer
        fs = FluidSynth(SoundFont_path)

        # Save the output to a WAV file
        wav_file = os.path.join(output_dir, f'{file_name}.wav')
        wav_file = wav_file.replace('\\', '/')
        
        fs.midi_to_audio(midi_file, wav_file)

if __name__ == "__main__": 
    
    SoundFont_path = "FluidR3_GM/FluidR3_GM.sf2"
    midi_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/2000/midi"    
    output_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/2000/wavs"
    
    midi_2_audio(SoundFont_path, midi_dir, output_dir)