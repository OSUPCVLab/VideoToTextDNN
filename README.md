# MTLE

Official code release for our 1st place submission to the LSMDC17 video-to-text competition.

#### NOTE: This is the debug branch, expect missing/changing information as we work on finalizing the release. 

Big thanks to Li Yao and his original project [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid), which this project derives from. 

## Dependencies

These are the general, high-level dependencies:

- CUDA-capable GPU(s)
- Large storage medium for dataset videos (for re-creating results)
- Python 2.7 + Python 3.5 (both required, more info below)

For Python 2.7:

- `Theano 0.8.1`
    - `cuDNN 5.4`
    - `CNMeM Memory backend` (optional)
    - Working `.theanorc` config file (provided, more info below)

For Python 3.5 and above:

- `PyTorch 0.4.0` (torch + torchvision)
- `pretrainedmodels` (Cadene repository)

In-depth Python package information is provided for each respective environment:

- `py2-vid-desc_requirements.txt`
- `py2_pip_freeze.txt`
- `py3-vid-desc_requirements.txt`
- `py3_pip_freeze.txt`

The rest of this guide assumes you are using Linux. 

We recommend the use of Anaconda to handle dependencies, as the files above can be used to easily re-create the necessary environments.

Theano uses a file called `.theanorc` to configure certain options. This file goes in your home directory on Linux. We have provided one that we use on a working test system, called `.theanorc.example`.  

#### Why two Python versions?

We use the `pretrainedmodels` package provided by GitHub user Cadene, due to its ease of use and better portability over Caffe.
However, this means having to use Python 3 for this specific step. Everything else uses Python 2.7. 
We thought this was a worthwhile hurdle to take advantage of PyTorch's ease of installation. 


## Installation

We recommend Anaconda, available [here](https://www.anaconda.com/download/).

Once Anaconda is installed, you must create two anaconda environments:

The general-purpose one:

`conda create --name vid-desc python=2.7 --file py2-vid-desc_requirements.txt`

and one for feature extraction:

`conda create --name vid-desc-feats python=3.6 --file py3-vid-desc_requirements.txt`

Use `conda activate <env-name>` to switch between environments. 


## Data

The data pipeline is handled under the `data/` directory. The `README.md` file there describes how to download the necessary datasets and process them for consumption in detail.

 

## Tutorial

Since the data collection process can take from minutes to weeks, depending on your available hardware, we have split the tutorial into two paths.

Visit the file `data/README.md` and follow the path most interesting to you to prepare the data. Once you have your data files ready, come back to this file to perform training or prediction on the path relevant to you.

The rest of this tutorial assumes you have either 1) extracted feature files, or 2) created `.pkl` files, as described in `data/README.md`. If there are any problems, feel free to file an issue on this repo, as this release is still a work in progress. 

####  "I just want to caption a couple of videos" (Prediction)


With your extracted features ready, you will need a pre-trained model. We have provided two checkpoint files, one trained on the 10k video MSVD Youtube-based dataset, and another on the 120k video LSMDC16 Movie dataset. 

MSVD Checkpoint: 

LSMDC Checkpoint:




#### "I  want to re-create your results" (Train, Evaluate, Prediction)

