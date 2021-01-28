# Copyright (C) 2020 Shahin Amiriparian, Maurice Gerczuk, Sandra Ottl, Björn Schuller
#
# This file is part of DeepSpectrum.
#
# DeepSpectrum is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSpectrum is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with DeepSpectrum. If not, see <http://www.gnu.org/licenses/>.
import os, yaml, subprocess, glob, csv
import pandas as pd
from tqdm import tqdm

params = {}
with open('params.yaml') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)
    params = params['opensmile']

feature_sets = params['featureSets']

# base directory for audio files
audio_base='./dist/wav'
output_base='./features/opensmile'
label_base='./dist/lab'
label_files = list(glob.glob(f'{label_base}/*.csv'))
print(label_files)
df = pd.read_csv(list(label_files)[0])
classes = sorted(set(df.label.values))
classtype = '{' + ','.join(classes) + '}'

for feature_set in feature_sets:
    out_dir = os.path.join(output_base, os.path.splitext(os.path.basename(feature_set))[0])
    os.makedirs(out_dir, exist_ok=True)
    for label_file in label_files:
        fid = [feature_set, os.path.basename(label_file)]
        featFile = os.path.join(*fid)
        outputFeat = os.path.join(output_base, featFile)
        outfile_base = os.path.join(out_dir, f'{os.path.splitext(os.path.basename(label_file))[0]}')
        outfile = f'{outfile_base}.arff'
        outfile_lld = f'{outfile_base}_lld.arff'
        with open(label_file) as f:
            reader = csv.reader(f, delimiter=',')
            next(reader) # skip header
            for filename, label in tqdm(reader):
                cmd = f'./opensmile/bin/SMILExtract -noconsoleoutput -C ./opensmile/config/{feature_set} -I {audio_base}/{filename} -N {filename} -class {label} -classtype "{classtype}" -O {outfile} -lldarffoutput {outfile_lld} -timestamparfflld 0'
                os.system(cmd)