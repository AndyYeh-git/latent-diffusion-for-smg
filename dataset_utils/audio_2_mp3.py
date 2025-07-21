import os
import glob
from pydub import AudioSegment

def audio_2_mp3(wav_dir, output_dir):

    os.makedirs(output_dir, exist_ok = True)
    wav_files = glob.glob(os.path.join(wav_dir, '*.wav'))
    for wav_file in wav_files:
        wav_file = wav_file.replace('\\', '/')
        file_name = wav_file.split('/')[-1][:-4]

        # Initialize the FluidSynth synthesizer
        sound = AudioSegment.from_wav(wav_file)
        
        # Save the output to a WAV file
        mp3_file = os.path.join(output_dir, f'{file_name}.mp3')
        mp3_file = mp3_file.replace('\\', '/')
        
        sound.export(mp3_file, format="mp3")
        print(f"✅ 轉換完成：{mp3_file}")



if __name__ == "__main__": 

    wav_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/2000/wavs/trimmed"    
    output_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/compose_with_me/finetuned/2000/mp3/trimmed"
    
    audio_2_mp3(wav_dir, output_dir)