# MTLE

This is the latest version of our code described in our [paper](https://arxiv.org/abs/1809.07257). An earlier version of our code was used at LSMDC17 where we won the movie description task. 

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

Clone the code as follows:

` git clone https://github.com/OSUPCVLab/VideoToTextDNN.git --recursive`

Once Anaconda is installed, you must create two anaconda environments:

The general-purpose one:

`conda create --name vid-desc python=2.7 --file py2-vid-desc_requirements.txt`

and one for feature extraction:

`conda create --name vid-desc-feats python=3.6 --file py3-vid-desc_requirements.txt`

Use `conda activate <env-name>` to switch between environments. 

Clone the repo recursively in order to clone other submodules that are required for the project

`git clone https://github.com/OSUPCVLab/VideoToTextDNN.git --recursive`

Install the required packages for the project

`pip install -r py2_pip_freeze.txt`

You might see some complaints about the following packages so you will need them to install them manually:

`conda install -c conda-forge pyro4`

For client installation, the following modules are also needed:

`python -m pip install --upgrade mss`

`conda install -c https://conda.anaconda.org/menpo opencv3`

`pip install pyttsx3`

`pip install pretrainedmodels`



## Data

The data pipeline is handled under the `data/` directory. The `README.md` file there describes how to download the necessary datasets and process them for consumption in detail.

 

## Tutorial

Since the data collection process can take from minutes to weeks, depending on your available hardware, we have split the tutorial into two paths.

Visit the file `data/README.md` and follow the path most interesting to you to prepare the data. Once you have your data files ready, come back to this file to perform training or prediction on the path relevant to you.

The rest of this tutorial assumes you have either 1) extracted feature files, or 2) created `.pkl` files, as described in `data/README.md`. If there are any problems, feel free to file an issue on this repo, as this release is still a work in progress. 


With your extracted features ready, you will need a pre-trained model. We have provided two checkpoint files, one trained on the 10k video MSVD Youtube-based dataset, and another on the 120k video LSMDC16 Movie dataset. 

MSVD and LSMDC Checkpoint: [download](https://uflorida-my.sharepoint.com/:f:/g/personal/w_garcia_ufl_edu/Ev7InIZkYc5Pn91wlU3oK1gB_NQ6BAArSll4iFELl8Hj2w?e=vad0K7)


## Demo
For a live demo, we make use of a server running our system and a client extracting and submiting cnn features to the server.

To start the server just run the script: 

 `python live_mtle_server.py <path_to_checkpoint>`
 
 The server then will output a temp_uri string that you need to use on the client to point where you want to send the input to.
 
To run the client just execute the following script with the mode you want to run the client. There are three types: live (screen capture), prompt (you pass the path of the video), headless (you pass a list of videos to process)

`python live_mtle_client.py <tem_uri> --mode <run_mode>`


## Acknowledgements
Big thanks to Li Yao and his original project [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid), which this project derives from. 

This work was sponsored by the SMART DOD program.

We apologize for the delay in releasing the code. The main author encountered some difficulties and life events leading up to the public release of the paper which made it difficult to relase the paper and code sooner. 
