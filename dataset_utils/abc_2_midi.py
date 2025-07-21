import os
import glob
import time

from music21 import converter, midi, stream, meter

def abc_2_midi(abc_dir, midi_dir):

    os.makedirs(midi_dir, exist_ok = True)
    success, success1 = 0, 0
    start_time = time.time()
    # Read the ABC notation file
    abc_files = glob.glob(os.path.join(abc_dir, '*.abc'))
    for abc_file in abc_files:
        abc_file = abc_file.replace('\\', '/')
        # Convert it to a MIDI file
        midi_file = os.path.join(midi_dir, f'{abc_file.split("/")[-1][:-4]}.mid')
        midi_file = midi_file.replace('\\', '/')                
        try:
            score = converter.parse(abc_file)
            mf = midi.translate.music21ObjectToMidiFile(score)
            mf.open(midi_file, 'wb')
            mf.write()
            mf.close()
            #success += 1
            #print(f'MIDI file saved as {midi_file}')
        except Exception as error:            
            #print(f'First {abc_file} failed due to {type(error).__name__} - {error}')
            try:
                score = converter.parse(abc_file)
                new_score = stream.Score()
                for part in score.parts:
                    new_part = stream.Part()
                    existing_time_signatures = set()
                    for element in part.flat.notesAndRests:
                        if isinstance(element, meter.TimeSignature):
                            if element not in existing_time_signatures:
                                existing_time_signatures.add(element)
                                new_part.append(element)
                        else:
                            new_part.append(element)
                    new_score.append(new_part)
                mf = midi.translate.music21ObjectToMidiFile(new_score)
                mf.open(midi_file, 'wb')
                mf.write()
                mf.close()
                #success1 += 1
                #print(f'MIDI file saved as {midi_file}')
            except Exception as error:
                pass
                #print(f'Second {abc_file} failed due to {type(error).__name__} - {error}')
    end_time = time.time()
    #print(f'Success MIDI file in First try {success}')
    #print(f'Success MIDI file total  {success + success1}')
    print(f'Convert time : {end_time - start_time} second')

if __name__ == "__main__":

    abc_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/laplace/infill_theory/abc/beam"
    midi_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/laplace/infill_theory/midi/beam"

    abc_2_midi(abc_dir, midi_dir)