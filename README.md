# MorFE

MorFE is a deep learning tool for analysis of cellular features for drug discovery.

The project presentation slides with an explanation and results can be found [here](https://bit.ly/MorFE-slides).

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

Create a virtual python environment with the libraries listed in the `build/requirements.txt` file. The original module was built using `python=3.7.5`.

The python environment can be built using `pyenv-virtualenv`:

    pyenv install 3.7.5
    pyenv virtualenv 3.7.5 MorFE
    pyenv activate MorFE
    pip install -r build/requirements.txt

Or alternatively, using `conda`:

    conda create -n MorFE python=3.7.5
    conda activate MorFE
    pip install -r build/requirements.txt

If using conda, make sure you use the correct `pip` installed within your `conda` virtual environment. If Anaconda3 is installed in the home directory, the correct pip can also be run with `~/anaconda3/envs/insight/bin/pip`.

#### 2.3 Dataset

MorFE was trained and validated using the [Broad Biomage Benchmark Collection #22 Dataset](https://data.broadinstitute.org/bbbc/BBBC022/). The dataset is comprised of a [metadata file](https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_image.csv) and an archive of [image files](https://data.broadinstitute.org/bbbc/BBBC022/BBBC022_v1_images_urls.txt).

> Describe the Dataset

##### 2.3.1 Download

The dataset can be downloaded for local use by running the following script. The script should:

-   Download the metadata file
-   Download the compressed zip files
-   Extract the zipped images

As the dataset is quite large (~1.5 TB) this will take a significant amount of time. In addition, the data must be stored on a large enough hard drive. If working on an EC2 instance, it is recommend to *mount an external volume* for local data storage. The directory or mount point where the dataset will be stored should be passed to the bash script:

    bash build/download_dataset.sh /path/to/data/drive

##### 2.3.2 Clean the metadata

Unfortunately, some of the zipped files are corrupted or missing. This leaves the metadata file full up entries that point to dead paths. In order to prune invalid samples from the dataset and clean the metadata headers, run the following command:

    python build/clean_metadata.py

## 3. Web App

    streamlit webapp.py

## 4. Command Line Tool

### 4.1 Inference

### 4.2 Training
