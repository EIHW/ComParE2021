# Baseline scripts for ComParE2021

Information on each sub-challenge is given in the respective branch. 
```
CCS - Covid Coughing Sub-challenge
CSS - Covid Speech Sub-challenge
PRS - Primates Sub-challenge
ESS - Escalation Sub-challenge 
```

## General Installation
### Linux
If you have conda installed (either miniconda or anaconda), you can execute `./install` to setup the two virtual environments needed for executing the experiments. You can activate the `core` or `end2you` environments with `source ./activate core` or `source ./activate end2you` respectively. 

### Installing openSMILE
openSMILE is needed for the extraction of LLDs and audio functionals. For Linux users, we provide a script (`scripts/download_opensmile.sh`) which downloads the latest binary release and puts it in the location expected by extraction codes. If you are on another platform, download and unpack the correct [binary release](https://github.com/audeering/opensmile/releases/tag/v3.0.0) to `opensmile` at the root of this repository. The layout should then look like this:
```
opensmile/
  |-- bin/
  |-- config/
  |-- ...
```
## Data
Make sure, you are on the correct branch regarding your chosen sub-challenge. Otherwise (with one of the virtual environments activated), checkout the desired branch. Move or copy the data from the challenge package into the project's root directory, such that the `dist` folder lies on the same level as `scripts/` and `src`. The layout should look like this:
```
dist/
  |-- wav/
  |-- lab/
  |-- features/
  |-- end2you/
```
Then, with one of the environments activated, run `dvc add dist`. The data will then be added to a local cache using `dvc`.

## Source code
`src` contains the python scripts used to run the baseline pipeline. It includes scripts for feature extraction and running the linear SVM classification.

## Scripts
We provide bash scripts for extracting auDeep features and running `end2you` on the data in `scripts/`.

## Reproducing the baseline results
If you installed the conda environments defined in `.env-ymls/`, you can reproduce the whole experimental pipeline from scratch by making use of [dvc](https://dvc.org/). Running `dvc repro` will execute the steps defined in `dvc.yaml` with parameters from `params.yaml`. `dvc` will detect changes to the inputs of any experimental stage and subsequent calls to `dvc repro` will rerun all affected steps. You can also check the current status of the pipeline with `dvc status`. If you want to run individual steps manually, you can also inspect `dvc.yaml` which defines the individual steps of the baseline experiments.

