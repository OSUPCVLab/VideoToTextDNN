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
 
Here the pipeline will be described in detail following the two paths established in the root `README.md`.  
 
 
## Tutorial
 
####  "I just want to caption a couple of videos" 






#### "I want to re-create your results" (MSVD & LSMDC)

 

