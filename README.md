# MorFE

MorFE is a deep learning tool for analysis of cellular features for drug
discovery.

The project presentation slides with an explanation and results can be found
[here](https://bit.ly/MorFE-slides).

## 1. Repository Structure

    MorFE
    ├── build
    │   └── Environment and data management scripts
    ├── configs
    │   └── Config files for command-line tool
    ├── data
    │   └── Metadata (.csv) file
    │   └── Image collections
    │       └── Image channels (.tiff)
    │            ...
    │        ....
    ├── train
        └── Code for retraining the model

## 2. Installation

#### 2.1 MorFE Repository

Checkout the repo using git:

    git clone https://github.com/jkhebel/MorFE
    cd MorFE

#### 2.2 Python Environment

Create a virtual python environment with the libraries listed in the
`build/requirements.txt` file. The original module was built using
`python=3.7.5`.

The python environment can be built using `pyenv-virtualenv`:

    pyenv install 3.7.5
    pyenv virtualenv 3.7.5 MorFE
    pyenv activate MorFE
    pip install -r build/requirements.txt

Or alternatively, using `conda`:

    conda create -n MorFE python=3.7.5
    conda activate MorFE
    pip install -r build/requirements.txt

If using conda, make sure you use the correct `pip` installed within your `
conda` virtual environment. If Anaconda3 is installed in the home directory,
the correct pip can also be run with `~/anaconda3/envs/insight/bin/pip`.

#### 2.3 Dataset

MorFE was trained and validated using the
[Broad Biomage Benchmark Collection #22 Dataset](https://data.broadinstitute.org/bbbc/BBBC022/). The dataset is comprised of a
[metadata file](https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv)
and an archive of
[image files](https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_urls.txt).

##### 2.3.1 Download

The dataset can be downloaded for local use by running the following script.
The script will:

-   Download the metadata file
-   Download the compressed zip files
-   Extract the zipped images

As the dataset is quite large (~1.5 TB) this will take a significant amount
of time (>24 hours). In addition, the data must be stored on a large enough hard drive.
If working on an EC2 instance, it is recommend to *mount an external volume*
for local data storage. The directory or mount point where the dataset will be
stored should be passed to the bash script:

    bash build/download_dataset.sh /path/to/data/drive

##### 2.3.2 Clean the metadata

Unfortunately, some of the zipped files are corrupted or missing.
This leaves the metadata file full up entries that point to dead paths.
In order to prune invalid samples from the dataset and clean the metadata
headers, run the following command:

    python build/clean_metadata.py

This script will generate a refined `dataset.csv` metadata file in the data directory.
It will also create a `cytotoxic.csv` metadata file, containing only cytotoxic
and control samples.

## 3. Demo

The demo script can be run locally using [Streamlit](streamlit.io):

    streamlit demo.py

If the demo is not automatically opened in your browser, open a new web page
and direct it to `http://localhost:8501`.

## 4. Command Line Tool

The CLI tool can be used to extract latent feature maps from the input samples,
or to train a new model for feature extraction. Run the python file with the
`--help` option to see usage instructions.

    python MorFE.py --help

Before declaring which `function` you'd like to run, you can first load a
configuration file. Configuration files are useful for preserving parameters
across multiple runs or seperate functions \(e.g. first training a model, then
extracting features with the same model\). Keep in mind that parameters defined
in the config file are later overwritten by any command-line arguents passed.

You can see the default arguments by examining the `configs/default.yml` file,
or supply your own config file using the `--config` option.

    python MorFE.py --config /path/to/config_file.yml function-name

By defaulte, MorFE loads the dataset defined by the metadata file stored at
`data/dataset.csv`. If the file is in a different directory, or you wish to
load a different dataset, the filepath can be passed using the `--dataset`
argument.

    python MorFE.py --dataset /path/to/dataset.csv function-name

Currently the following functions are implement:

- `extract-features` - use a pre-trained model to extract image features and predict cell organization.  
- `train` - train a new model for feature extraction using a provided dataset

### 4.1 Feature Extraction (Inference)

You can use the `extract-features` function to predict cell organization maps
from input samples, and extract the corresponding feature maps.

    python MorFe.py --dataset /path/to/dataset.csv extract-features

### 4.2 Training

If you would like to train your own model, this can be achieved using the `train`
function.

    python MorFe.py --dataset /path/to/dataset.csv train

## 5. Future Development

The following includes a list of future development tasks for this project:

 - Dockerization for easier deployment
 - Implement a single build script that utilizes `setuptools`
 - Feature extraction and classificaiton using segmented cells
