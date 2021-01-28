import numpy as np
import collections
import argparse
import sys

from pathlib import Path
from moviepy.editor import AudioFileClip


parser = argparse.ArgumentParser(description='Covert Labels to End2You format -- flags.')
parser.add_argument('--covid_path', type=str, 
                    default='./e2u_output',
                    help='Path to COVID data.')
parser.add_argument('--save_path', type=str, 
                    default='./e2u_output',
                    help='Path to save End2You conversion files.')

fps = 16000
win_len = 0.1
audio_chunk = int(win_len*fps)

def convert_files(covid_path, partition, save_path):
    label_path = covid_path / 'lab' / f'{partition}.csv'
    audio_files_path = covid_path / 'wav'
    
    labels = np.loadtxt(str(label_path), dtype=str, delimiter=',', skiprows=1)
    
    file, label  = labels[:,0], labels[:,1]
    
    for i, l in enumerate(np.unique(label)):
        idx = np.where(label == l)
        label[idx] = i
    
    save_path.mkdir(exist_ok=True)
    (save_path / 'labels').mkdir(exist_ok=True)
    
    input_files = [] 
    for f, l in zip(*[file, label]):
        
        audio_file = audio_files_path / f
        if not audio_file.exists():
            continue
        
        print(f'Writing csv file for: [{f}]')
        clip = AudioFileClip(str(audio_file),fps=fps)
        num_samples = int(fps*clip.duration // audio_chunk)
        
        file_data = []
        for i in range(num_samples+1):
            file_data.append([round(i*win_len, 2), l])
        data_array = np.array(file_data)
        
        save_label_file_path = save_path / 'labels' / (f[:-4] + '.csv')
        np.savetxt(str(save_label_file_path), data_array, delimiter=',', header='file,label', fmt='%s')
        
        input_files.append([str(audio_files_path / f), str(save_path / 'labels' / (f[:-4] + '.csv'))])
        
        # Upsample for training
        if 'train' in f and l == 1:
            np.savetxt(str(save_path / 'labels' / (save_label_file_path.name[:-4] + '_2.csv')), data_array, delimiter=',', header='file,label', fmt='%s')
            np.savetxt(str(save_path / 'labels' / (save_label_file_path.name[:-4] + '_3.csv')), data_array, delimiter=',', header='file,label', fmt='%s')
            input_files.append([str(audio_files_path / f), str(save_path / 'labels' / (f[:-4] + '_2.csv'))])
            input_files.append([str(audio_files_path / f), str(save_path / 'labels' / (f[:-4] + '_3.csv'))])
    
    input_file_path = save_path / 'labels' / f'{partition}_input_file.csv'
    np.savetxt(str(input_file_path), np.array(input_files), delimiter=',', fmt='%s', header='raw_file,label_file')


if __name__ == '__main__':
    flags = sys.argv[1:]
    flags = vars(parser.parse_args(flags))
    covid_path, save_path = Path(flags['covid_path']), Path(flags['save_path'])
    
    for partition in ["train", "devel", "test"]:
        convert_files(covid_path, partition, save_path)
    