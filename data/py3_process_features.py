import sys
import torch
import pretrainedmodels
import pretrainedmodels.utils as utils
import torch.nn as nn
import argparse
import time
import data.validate_feats as validate_feats
import os
import numpy as np
import logging
import shutil

from PIL import Image
from multiprocessing import Pool

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

available_features = ['nasnetalarge', 'resnet152', 'pnasnet5large', 'densenet121', 'senet154', 'polynet']

args = None


def init_model(gpu_ids, model_name, fromfile=True):
    # model_name = 'pnasnet5large'
    # could be fbresnet152 or inceptionresnetv2
    model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
    model.eval()

    if fromfile:
        load_img = utils.LoadImage()
    else:
        class LoadOpenCVImage(object):

            def __init__(self, space='RGB'):
                self.space = space

            def __call__(self, cv_img):
                pil_im = Image.fromarray(cv_img)
                pil_im = pil_im.convert(self.space)
                return pil_im

        load_img = LoadOpenCVImage()

    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)

    """
    TODO(WG): Would be nice to use something like DataParallel, but that only does forward pass on given module.
    Need to stop before logits step. 
    Should create wrapper for pretrainedmodels that does the MPI-like ops across GPUs on model.features modules:
    1) replicated
    2) scatter
    3) parallel_apply
    4) gather
    Would have to know what layers are being used on each model. 
    """
    if torch.cuda.is_available():
        model = model.cuda(device=gpu_ids[0])

    return load_img, tf_img, model


def extract_features(args):
    root_frames_dir = args.frames_dir
    root_feats_dir = args.feats_dir
    work = args.work
    autofill = int(args.autofill)
    ftype = args.type
    gpu_list = args.gpu_list

    frames_dirs = os.listdir(root_frames_dir)

    if not os.path.isdir(root_feats_dir):
        os.mkdir(root_feats_dir)
    # else:
    #     if autofill:
    #         logger.info('AUTOFILL ON: Attempting to autofill missing features.')
    #         frames_dirs = validate_feats.go(featsd=root_feats_dir, framesd=root_frames_dir)

    # Difficulty of each job is measured by # of frames to process in each chunk.
    # Can't be randomized since autofill list woudld be no longer valid.
    # np.random.shuffle(frames_dirs)
    work = len(frames_dirs) if not work else work

    load_img, tf_img, model = init_model(args.gpu_list, args.type)

    work_done = 0
    while work_done != work:
        frames_dirs_avail = diff_feats(root_frames_dir, root_feats_dir)
        if len(frames_dirs_avail) == 0:
            break

        frames_dir = np.random.choice(frames_dirs_avail)
        ext = '.' + frames_dir.split('.')[-1]
        feat_filename = frames_dir.split('/')[-1].split(ext)[0]
        video_feats_path = os.path.join(args.feats_dir, feat_filename)

        if os.path.exists(video_feats_path):
            logger.info('Features already extracted:\t{}'.format(video_feats_path))
            continue

        try:
            frames_to_do = [os.path.join(args.frames_dir, frames_dir, p) for p in
                            os.listdir(os.path.join(args.frames_dir, frames_dir))]
        except Exception as e:
            logger.exception(e)
            continue

        # Must sort so frames follow numerical order. os.listdir does not guarantee order.
        frames_to_do.sort()

        if len(frames_to_do) == 0:
            logger.warning("Frame folder has no frames! Skipping...")
            continue

        # Save a flag copy
        with open(video_feats_path, 'wb') as pf:
            np.save(pf, [])

        try:
            batches = create_batches(frames_to_do, load_img, tf_img, batch_size=args.batch_size)
        except OSError as e:
            logger.exception(e)
            logger.warning("Corrupt image file. Skipping...")
            os.remove(video_feats_path)
            continue

        logger.debug("Start video {}".format(work_done))

        feats = process_batches(batches, ftype, gpu_list, model)

        with open(video_feats_path, 'wb') as pf:
            np.save(pf, feats)
            logger.info('Saved complete features to {}.'.format(video_feats_path))
        work_done += 1


def process_batches(batches, ftype, gpu_list, model):
    done_batches = []
    for i, batch in enumerate(batches):
        if torch.cuda.is_available():
            batch = batch.cuda(device=gpu_list[0])

        output_features = model.features(batch)
        output_features = output_features.data.cpu()

        conv_size = output_features.shape[-1]

        if ftype == 'nasnetalarge' or ftype == 'pnasnet5large':
            relu = nn.ReLU()
            rf = relu(output_features)
            avg_pool = nn.AvgPool2d(conv_size, stride=1, padding=0)
            out_feats = avg_pool(rf)
        else:
            avg_pool = nn.AvgPool2d(conv_size, stride=1, padding=0)
            out_feats = avg_pool(output_features)

        out_feats = out_feats.view(out_feats.size(0), -1)
        logger.info('Processed {}/{} batches.\r'.format(i + 1, len(batches)))

        done_batches.append(out_feats)
    feats = np.concatenate(done_batches, axis=0)
    return feats


def create_batches(frames_to_do, load_img_fn, tf_img_fn, batch_size=8):
    n = len(frames_to_do)
    if n < batch_size:
        logger.warning("Sample size less than batch size: Cutting batch size.")
        batch_size = n

    logger.info("Generating {} batches...".format(n // batch_size))
    batches = []
    frames_to_do = np.array(frames_to_do)

    for idx in range(0, n, batch_size):
        frames_idx = list(range(idx, min(idx+batch_size, n)))
        batch_frame_refs = frames_to_do[frames_idx]

        batch_tensor = torch.zeros((len(batch_frame_refs),) + tuple(tf_img_fn.input_size))
        for i, frame_ref in enumerate(batch_frame_refs):
            input_img = load_img_fn(frame_ref)
            input_tensor = tf_img_fn(input_img)  # 3x400x225 -> 3x299x299 size may differ
            # input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
            batch_tensor[i] = input_tensor

        batch_ag = torch.autograd.Variable(batch_tensor, requires_grad=False)
        batches.append(batch_ag)

    return batches


def diff_feats(frames_dir, feats_dir):
    feats = set(os.listdir(feats_dir))
    frames_to_ext = {'.'.join(i.split('.')[:-1]): i.split('.')[-1] for i in os.listdir(frames_dir)}
    frames = set(frames_to_ext.keys())
    needed_feats = frames - feats
    needed_feats = [i + '.' + frames_to_ext[i] for i in needed_feats]
    return needed_feats


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('frames_dir',help = 'Directory where there are frame directories.')
    arg_parser.add_argument('feats_dir',help = 'Root directory of dataset\'s processed videos.')
    arg_parser.add_argument('-w', '--work', help = 'Number of features to process. Defaults to all.', default=0, type=int)
    arg_parser.add_argument('-gl', '--gpu_list', required=True, nargs='+', type=int, help="Space delimited list of GPU indices to use. Example for 4 GPUs: -gl 0 1 2 3")
    arg_parser.add_argument('-bs', '--batch_size', type=int, help="Batch size to use during feature extraction. Larger batch size = more VRAM usage", default=8)
    arg_parser.add_argument('--type', required=True, help = 'ConvNet to use for processing features.', choices=available_features)
    arg_parser.add_argument('--autofill', action='store_true', default=False, help="Perform diff between frames_dir and feats_dir and fill them in.")

    args = arg_parser.parse_args()

    start_time = time.time()

    logger.info("Found {} GPUs, using {}.".format(torch.cuda.device_count(), len(args.gpu_list)))

    extract_features(args)

    logger.info("Job took %s mins" % ((time.time() - start_time)/60))
