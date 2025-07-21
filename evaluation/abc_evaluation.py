import os
import glob
import numpy as np
import math
import muspy

def calculate_mean_std(data):
    """
    Calculates the mean and standard deviation of a dictionary where keys are numbers
    and values are their appearances.

    Args:
        data: A dictionary where keys are numbers and values are their counts.

    Returns:
        A tuple containing the mean and standard deviation. Returns (None, None) if the
        dictionary is empty.
    """
    if not data:
        return None, None, None

    total_count = sum(data.values())
    if total_count == 0:
        return 0, None, None

    mean = sum(key * count for key, count in data.items()) / total_count

    variance = sum((key - mean)**2 * count for key, count in data.items()) / total_count
    std_dev = math.sqrt(variance)

    return total_count, mean, std_dev

def evaluate_pitch_classes_used(abc_dir):
    abc_files = glob.glob(os.path.join(abc_dir, '*.abc'))
    total_pitch_classes_used = 0.0
    pitch_classes_used_dic = {}
    error_files = 0
    step = 0
    for abc_file in abc_files:
        abc_file = abc_file.replace('\\', '/')
        if step % 5000 == 0:
            print(f'Step {step} : {abc_file}')
        step += 1
        try:
            music = muspy.read(abc_file)
            pitch_classes_used = muspy.n_pitch_classes_used(music)
            if np.isnan(pitch_classes_used):
                error_files += 1
            else:
                total_pitch_classes_used += pitch_classes_used
                if pitch_classes_used not in pitch_classes_used_dic:
                    pitch_classes_used_dic[pitch_classes_used] = 1
                else:
                    pitch_classes_used_dic[pitch_classes_used] += 1
        except Exception as error:
            error_files += 1            
            # print(f'{abc_file} failed due to {type(error).__name__} - {error}')

    total_count, mean, std_dev = calculate_mean_std(pitch_classes_used_dic)
    print(f'Total files in {abc_dir} : {len(abc_files)}')
    print(f'Error files in {abc_dir} : {error_files}')
    print(f'Suceessful files in {abc_dir} : {total_count}')
    print(f'Average pitch_classes_used at {abc_dir} : {total_pitch_classes_used / (len(abc_files) - error_files)}')
    print(f'Error rate {abc_dir} : {float(error_files) / len(abc_files)}')
    print(f'Suceessful files Mean in {abc_dir} : {mean}')
    print(f'Suceessful files STD in {abc_dir} : {std_dev}')

def evaluate_pitch_entropy(abc_dir):
    abc_files = glob.glob(os.path.join(abc_dir, '*.abc'))
    total_pitch_entropy = 0.0
    pitch_entropy_dic = {}
    error_files = 0
    step = 0
    for abc_file in abc_files:
        abc_file = abc_file.replace('\\', '/')
        if step % 5000 == 0:
            print(f'Step {step} : {abc_file}')
        step += 1
        try:
            music = muspy.read(abc_file)
            pitch_entropy = muspy.pitch_entropy(music)
            if np.isnan(pitch_entropy):
                error_files += 1
            else:
                total_pitch_entropy += pitch_entropy
                if pitch_entropy not in pitch_entropy_dic:
                    pitch_entropy_dic[pitch_entropy] = 1
                else:
                    pitch_entropy_dic[pitch_entropy] += 1
        except Exception as error:
            error_files += 1            
            # print(f'{abc_file} failed due to {type(error).__name__} - {error}')

    total_count, mean, std_dev = calculate_mean_std(pitch_entropy_dic)
    print(f'Total files in {abc_dir} : {len(abc_files)}')
    print(f'Error files in {abc_dir} : {error_files}')
    print(f'Suceessful files in {abc_dir} : {total_count}')
    print(f'Average pitch_entropy at {abc_dir} : {total_pitch_entropy / (len(abc_files) - error_files)}')
    print(f'Error rate {abc_dir} : {float(error_files) / len(abc_files)}')
    print(f'Suceessful files Mean in {abc_dir} : {mean}')
    print(f'Suceessful files STD in {abc_dir} : {std_dev}')

def evaluate_pitch_class_entropy(abc_dir):
    abc_files = glob.glob(os.path.join(abc_dir, '*.abc'))
    total_pitch_class_entropy = 0.0
    pitch_class_entropy_dic = {}
    error_files = 0
    step = 0
    for abc_file in abc_files:
        abc_file = abc_file.replace('\\', '/')
        if step % 5000 == 0:
            print(f'Step {step} : {abc_file}')
        step += 1
        try:
            music = muspy.read(abc_file)
            pitch_class_entropy = muspy.pitch_class_entropy(music)
            if np.isnan(pitch_class_entropy):
                error_files += 1
            else:
                total_pitch_class_entropy += pitch_class_entropy
                if pitch_class_entropy not in pitch_class_entropy_dic:
                    pitch_class_entropy_dic[pitch_class_entropy] = 1
                else:
                    pitch_class_entropy_dic[pitch_class_entropy] += 1
        except Exception as error:
            error_files += 1            
            # print(f'{abc_file} failed due to {type(error).__name__} - {error}')

    total_count, mean, std_dev = calculate_mean_std(pitch_class_entropy_dic)
    print(f'Total files in {abc_dir} : {len(abc_files)}')
    print(f'Error files in {abc_dir} : {error_files}')
    print(f'Suceessful files in {abc_dir} : {total_count}')
    print(f'Average pitch_class_entropy at {abc_dir} : {total_pitch_class_entropy / (len(abc_files) - error_files)}')
    print(f'Error rate {abc_dir} : {float(error_files) / len(abc_files)}')
    print(f'Suceessful files Mean in {abc_dir} : {mean}')
    print(f'Suceessful files STD in {abc_dir} : {std_dev}')

def evaluate_groove_consistency(abc_dir):
    abc_files = glob.glob(os.path.join(abc_dir, '*.abc'))
    total_groove_consistency = 0.0
    groove_consistency_dic = {}
    error_files = 0
    step = 0
    for abc_file in abc_files:
        abc_file = abc_file.replace('\\', '/')
        if step % 5000 == 0:
            print(f'Step {step} : {abc_file}')
        step += 1
        try:
            music = muspy.read(abc_file)
            groove_consistency = muspy.groove_consistency(music, 16)
            if np.isnan(groove_consistency):
                error_files += 1
            else:
                total_groove_consistency += groove_consistency
                if groove_consistency not in groove_consistency_dic:
                    groove_consistency_dic[groove_consistency] = 1
                else:
                    groove_consistency_dic[groove_consistency] += 1
        except Exception as error:
            error_files += 1            
            # print(f'{abc_file} failed due to {type(error).__name__} - {error}')

    total_count, mean, std_dev = calculate_mean_std(groove_consistency_dic)
    print(f'Total files in {abc_dir} : {len(abc_files)}')
    print(f'Error files in {abc_dir} : {error_files}')
    print(f'Suceessful files in {abc_dir} : {total_count}')
    print(f'Average groove_consistency at {abc_dir} : {total_groove_consistency / (len(abc_files) - error_files)}')
    print(f'Error rate {abc_dir} : {float(error_files) / len(abc_files)}')
    print(f'Suceessful files Mean in {abc_dir} : {mean}')
    print(f'Suceessful files STD in {abc_dir} : {std_dev}')

if __name__ == "__main__":

    #abc_dir = "../irishman_abc/L1_8_M4_4_seg_8_strip_min_3_final/train"
    abc_dir = "../prediction/latent-diffusion-for-smg-final/irishman/L1_8_M4_4_seg_8_strip_min_3/laplace/infill_theory/abc/beam"

    evaluate_pitch_classes_used(abc_dir)
    evaluate_pitch_entropy(abc_dir)
    evaluate_pitch_class_entropy(abc_dir)
    evaluate_groove_consistency(abc_dir)
