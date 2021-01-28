import os, yaml
import pandas as pd

params = {}
with open('params.yaml') as f:
    p = yaml.load(f, Loader=yaml.FullLoader)
    params = p['XBOW']

conf_attributes_map = {'ComParE_2016': 'n1[65]2[65]c'}

# Paths
os_features_folder = 'features/opensmile'
features_folder = 'features/openXBoW'

os.makedirs(features_folder, exist_ok=True)

# Compute BoAW representations from openSMILE LLDs
for dir in os.listdir(os_features_folder):
    for csize in params['csize']:
        for num_assignments in params['num_assignments']:
            for part in ['train', 'devel', 'test']:
                output_dir = os.path.join(features_folder, dir, str(csize), str(num_assignments))
                output_file_boaw = os.path.join(output_dir, f'{part}.arff')
                os.makedirs(output_dir, exist_ok=True)
                if dir in conf_attributes_map:
                    xbow_config = f'-i {os.path.join(os_features_folder, dir, part + "_lld.arff")} -attributes {conf_attributes_map[dir]} -o {output_file_boaw}'
                    if part=='train':
                        xbow_config += f' -standardizeInput -size {csize} -a {num_assignments} -log -B {os.path.join(output_dir, "codebook")}'
                    else:
                        xbow_config += f' -b {os.path.join(output_dir, "codebook")}'
                    os.system('java -Xmx12000m -jar ./openxbow/openXBOW.jar -writeName ' + xbow_config)
                else:
                    print(f'{dir} not mapped to attribute config. Mapped confs: {conf_attributes_map}.')
