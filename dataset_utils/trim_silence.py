import os
import glob

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

def trim_silence(audio, silence_thresh=-50.0, padding=100):
    # Find non-silent range
    non_silence = detect_nonsilent(audio, min_silence_len=500, silence_thresh=silence_thresh)
    if not non_silence:
        return audio  # No non-silent section detected
    start_trim = max(0, non_silence[0][0] - padding)
    end_trim = min(len(audio), non_silence[-1][1] + padding)
    return audio[start_trim:end_trim]

def trim_silence_dir(input_dir, output_dir, silence_thresh=-50.0, padding=100):

    # Load and trim
    os.makedirs(output_dir, exist_ok = True)
    input_files = glob.glob(os.path.join(input_dir, '*.wav'))
    for input_file in input_files:
        print(f'input file : {input_file}')
        input_file = input_file.replace('\\', '/')
        file_name = input_file.split('/')[-1][:-4]
        sound = AudioSegment.from_wav(input_file)
        trimmed = trim_silence(sound)

        output_file = os.path.join(output_dir, f'{file_name}.wav')
        output_file = output_file.replace('\\', '/')
        trimmed.export(output_file, format="wav")

if __name__ == "__main__": 
    
    input_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/2000/wavs"
    output_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/2000/wavs/trimmed"
    
    trim_silence_dir(input_dir, output_dir)
