import os
import json
import glob
import time

def save_abc(json_path, abc_dir, source_path=None):

    os.makedirs(abc_dir, exist_ok = True)
    start_time = time.time()
    json_file = open(json_path, 'r', encoding='utf-8')
    datas = json.load(json_file)

    if source_path != None:
        source_file = open(source_path, 'r', encoding='utf-8')
        source_datas = json.load(source_file)
        for i, data in enumerate(datas):
            abc_data = source_datas[i].split(' ')
            abc_data.append(data)
            # Specify the output file path
            abc_file_path = os.path.join(abc_dir, f'{i}.abc')
            abc_file_path = abc_file_path.replace('\\', '/')

            # Write the ABC data to the file
            with open(abc_file_path, 'w') as abc_file:
                for line in abc_data:
                    abc_file.write(line + '\n')
                #print(f'ABC notation saved to {abc_file_path}')                    
    else:
        for key, value in datas.items():
            os.makedirs(os.path.join(abc_dir, key), exist_ok = True)
        for key, value in datas.items():
            for i, data in enumerate(value):
                # Your ABC notation data
                abc_data = [
                    "L:1/8",
                    "M:4/4"
                ]
                abc_data.append(data)
                # Specify the output file path
                abc_file_path = os.path.join(abc_dir, f'{key}/{i}.abc')
                abc_file_path = abc_file_path.replace('\\', '/')

                # Write the ABC data to the file
                with open(abc_file_path, 'w') as abc_file:
                    for line in abc_data:
                        abc_file.write(line + '\n')
                    #print(f'ABC notation saved to {abc_file_path}')
    end_time = time.time()
    print(f'Convert time : {end_time - start_time} second')

def save_inference_abc(json_path, abc_dir, source_path=None):

    os.makedirs(abc_dir, exist_ok = True)
    start_time = time.time()
    json_file = open(json_path, 'r', encoding='utf-8')
    datas = json.load(json_file)

    for key, value in datas.items():
        os.makedirs(os.path.join(abc_dir, key), exist_ok = True)
    for key, value in datas.items():
        for i, data in enumerate(value):
            # Your ABC notation data
            abc_data = [
                "L:1/8",
                "M:4/4"
            ]
            abc_data.append('|'.join(data.split('|')[2:6]))
            # Specify the output file path
            abc_file_path = os.path.join(abc_dir, f'{key}/{i}.abc')
            abc_file_path = abc_file_path.replace('\\', '/')

            # Write the ABC data to the file
            with open(abc_file_path, 'w') as abc_file:
                for line in abc_data:                        
                    abc_file.write(line + '\n')
            print(f'ABC notation saved to {abc_file_path}')
    end_time = time.time()
    print(f'Convert time : {end_time - start_time} second')

if __name__ == "__main__":

    json_path = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/laplace/infill_theory/42_val_prediction.json"
    abc_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/laplace/infill_theory/abc"
    source_path=None

    save_abc(json_path, abc_dir, source_path=source_path)
    #save_inference_abc(json_path, abc_dir, source_path=source_path)
