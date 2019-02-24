# `vid-desc/data`

## What is here?

Standalone scripts for generating evalation data from these datasets:

- MSR-VTT
- M-VAD
- MPII
- LSMDC 2016 (M-VAD + MPII)
- TRECVID 2016
- MSVD (Youtube2text)
- TACoS

Multi-caption datasets rely on `pickle` generated files to store features, while single-caption datasets do not.
This is mainly an artifact of the dataset sizes. Generating `pkl` files for LSMDC, MPII, and MVAD was usually unwieldy.

Due to the modular nature, this pipeline can also be used as a general-purpose feature extractor for any videos. 


## How are scripts run?

The general pipeline is this:

- `download videos to vid_dir` 
- `subsect_videos(vid_dir)`* 
- `process_frames(subsect_dir)`
- `process_features(frames_dir)`
- `create_dataset(feats_dir)`* 

`*` denotes a process that depends on dataset meta data. Everything else is dataset agnostic. 

Everything except `process_frames` uses the python2 environment. `process_frames` uses python3, which is a result of upgrading to pytorch for feature extraction. 
For the time being, you will just need to switch between two conda environments during the pipeline. 

So long as you have a directory with videos in it, the usage of each step should be clear. If you are not trying to re-create the results for some dataset, you can ommit `subsect_videos` and `create_dataset`, as those are for specific datasets.
 
Here the pipeline will be described in detail following path 2 established in the root `README.md`.    
 
 
## Tutorial (re-creating results)  

We assume successful installation from root README. You only need the Theano conda environment for this tutorial. 

Before proceeding, we must patch a small issue with coco-caption.

At `coco-caption/pycocoevalcap/bleu/bleu.py:35` replace

`            assert(len(ref) > 1)`
with
`            assert(len(ref) > 0)`

### MSVD


1) Download the checkpoint and pickle archives from the hosted link in project root's README.

2) Extract the pickle archive, `/path/to/msvd/pkls`.

3) Extract the checkpoint archive, `/path/to/msvd/ckpt/`

4) Switch to Theano conda environment 

5) ...

### LSMDC16

Note: You will need around 9GB of total VRAM to complete this. 


1) Download the checkpoint, feature, and pickle archives from the hosted link in project root's README.

2) Extract the downloaded features. For LSMDC, the large archive meant splitting to fit onto OneDrive. Concatenate the part files back together to get the original:

    `cat /path/to/lsmdc16/features_resnet.tar.gz.part* > /path/to/lsmdc16/features_resnet.tar.gz`
    
    Then extract it, `/path/to/lsmdc16/features_resnet/`

3) Extract the checkpoint directory somewhere. We will refer to it as `/path/to/lsmdc16/ckpt/`.

4) Extract the pickle directory somewhere, `/path/to/lsmdc16/pkls16/`

5) We're going to backup the model checkpoint and save the sampled test and validation samples to a directory, `/path/to/lsmdc16/scoring_files/`. Around 4GB needed. 

6) Switch to Theano conda environment 

7) Now, you can run the deceptively named `train_model.py` (from repo root) on validate mode to get some scores:

    ```bash
    THEANO_FLAGS='mode=FAST_RUN,device=gpu0,lib.cnmem=9000' \
    python train_model.py model='lstmdd' \
    random_seed=9999 \
    lstmdd.dataset='lsmdc16' \
    lstmdd.data_dir='/path/to/lsmdc16/pkls16/' \
    lstmdd.video_feature='resnet' \
    lstmdd.feats_dir='/path/to/lsmdc16/features_resnet/' \
    lstmdd.save_model_dir='path/to/lsmdc16/scored_files/' \
    lstmdd.validFreq=2000 \
    lstmdd.encoder='lstm_bi' lstmdd.cost_type='v1' lstmdd.dec='generative' \
    lstmdd.mode='validate' \
    lstmdd.reload_=True lstmdd.from_dir='/path/to/lsmdc16/ckpt/'
    ```
    
8) Scores will print out after some agonizing minutes. 

9) (optional) Samples are saved to `/path/to/lsmdc16/scoring_files/`. If you need the scores later, you can run it with the standalone `metrics.py` (from repo root) and get the scores without having to generate samples again:

    ```bash
    python metrics.py --model_type lstmdd --dataset lsmdc16 --ckpt_dir /path/to/lsmdc16/ckpt/ --data_dir /path/to/lsmdc16/pkls16/ --feature_dir /path/to/lsmdc16/features_resnet --caption_dir /path/to/lsmdc16/scoring_files/
    ``` 