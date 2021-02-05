# DiFE: Dutch lIngustic Feature Extractor
DiFE: Dutch lIngustic Feature Extractor is a simple Python pipeline to reproduce the extracted linguistic features from the ComParE2021 challenge. We utilise and provide contextual word embeddings using a frozen and fine-tuned Dutch Bidirectional Language Transformer (Bert).

# Installation

Python 3.6 or higher is required. The recommended type of installation is through `pip` in a separate virtualenv. 

### Python Installation

Assuming Ubuntu OS, execute the following steps:

1. Make sure that pip is up-to-date by running:
```
python -m pip install --user --upgrade pip
```
2. Install virtualenv
```
python -m pip install --user virtualenv
```
3. Create a new virtualenv som-e
```
python -m venv dife
```
4. Activate the created virtualenv
```
source dife/bin/activate
```
5. Install the packages according to requirements.txt - if GPU support is wished, the corresponding CUDA Toolkit etc. has to be installed first
```
pip install -r requirements.txt
```
6. Use the provided code & data or clone from github and add the data. 
```
data: xxx
features: xxx
```
7. Run all models using
```
python run_all.sh
```

### Python Dependencies (installed automatically during setup)
These Python packages are installed automatically during setup by `pip install -r requirements.txt`, and are just listed for completeness.

tensorflow==2.4.1
simpletransformers==0.51.15
torch==1.7.1
numpy==1.19.4

### Additional Comments for GPU Usage

The pipeline is designed so that the final features from the models, which are subsequently fed into the SVM, can be computed with a desktop computer in reasonable computing time. By default, T_DEVICE is switched to `cpu`, so if the Bert extraction is re-calculated, torch (simpletransformers) does not move the Tensors to the GPU (.cuda()). Any other parameter e.g. `gpu` activates GPU usage. 

If GPU support is available, Tensorflow/Keras will still use GPU(s) for the model training, otherwise falls back on CPU and the pipeline runs entirely on CPU.

We provide pre-computed contextual word embeddings extracted from a Dutch Bert for all partitions of `ComParE2021` to save computation time. They can also automatically recalculated. In this case, GPU support is recommended.

The compatible CUDA libraries (CUDA Toolkit, cuDNN) are required to be available on the system path (which should be the case after a standard installation). 

# Description

## General
All models include a sequence of words to segment vector mapping (e.g. by a global max pooling or a BiLSTM + Attention) followed by two 128-dim FF (relu) and one 128-dim FF (sigmoid) layer. The output of this final layer is treated as the feature input for the SVM evaluation. 
