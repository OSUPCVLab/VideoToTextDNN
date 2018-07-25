"""
Python3 script
"""

import os
import argparse
import logging
import Pyro4

from data.process_frames import process_vid
from data.py3_process_features import create_batches, process_batches, available_features, init_model

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mkdirs_safe(dir):
    try:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    except OSError as e:
        logger.exception(e)


def np_to_serpent(features):
    """
    Convert numpy float32 2d array to basic type compatible with serpent
    :param features:
    :return:
    """
    serp_features = []

    for feat in features:
        serp_feat = []
        for val in feat:
            serp_feat.append(float(val))

        serp_features.append(serp_feat)

    return serp_features


def watch(args):
    load_img, tf_img, feats_model = init_model(args.gpu_list, args.feature_type)

    uri = args.uri
    remote_captioner = Pyro4.Proxy(uri)

    modelling_refs = (load_img, tf_img, feats_model, remote_captioner)

    if not os.path.isdir(args.temp_dir):
        mkdirs_safe(args.temp_dir)

    # Where we store frames before we begin feature processing. TODO: Could be a tempfile.TemporaryDirectory instead
    root_frames_dir = os.path.join(args.temp_dir, 'frames')
    if not os.path.isdir(root_frames_dir):
        mkdirs_safe(root_frames_dir)

    try:
        if args.mode == 'headless':
            for vpath in args.videos:
                caption = vid_path_to_caption(root_frames_dir, vpath, modelling_refs)
                logger.info("Caption for [{}]: {}".format(vpath, caption))
        elif args.mode == 'live':
            option = ''
            while option.lower() != 'q':
                print("List of options:")
                print("q: quit, c: provide a video path, cc: provide video paths infinitely")
                option = input("-> ").lower().strip()
                if option == 'c':
                    vpath = input("Provide path to video: ").strip()
                    caption = vid_path_to_caption(root_frames_dir, vpath, modelling_refs)
                    print("CAPTION: " + caption)
                if option == 'cc':
                    vpath = ''
                    while vpath != 'q':
                        vpath = input("Provide path to video (q to exit): ").strip()
                        caption = vid_path_to_caption(root_frames_dir, vpath, modelling_refs)
                        print("CAPTION: " + caption)

    except Exception as e:
        logger.exception(e)


def vid_path_to_caption(root_frames_dir, vpath, modelling_refs):
    (load_img, tf_img, feats_model, remote_captioner) = modelling_refs

    vid_name = vpath.split('/')[-1]
    frames_dir = os.path.join(root_frames_dir, vid_name)

    if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) != 0:
        logger.warning("Frames already exist at {}".format(frames_dir))
    else:
        logger.info("Extracting frames...")
        process_vid(('', '', vpath, frames_dir))

    frames = [os.path.join(frames_dir, frame) for frame in os.listdir(frames_dir)]

    logger.info("Exracting features...")

    try:
        batches = create_batches(frames, load_img, tf_img, batch_size=8)
    except OSError as e:
        logger.exception(e)
        logger.warning("Corrupt image file. Skipping...")
        return "nil"

    feats = process_batches(batches, args.feature_type, args.gpu_list, feats_model)

    # print(feats[0][:5])

    logger.info("Extracted {} features.".format(len(feats)))

    logger.info("Captioning...")
    return remote_captioner.caption_features(np_to_serpent(feats))


def _validate(args):
    if args.mode == 'headless':
        if not args.videos:
            raise StandardError("You must use the --videos option for headless mode.")

        for vpath in args.videos:
            if not os.path.exists(vpath):
                raise IOError("No video exists at {}".format(vpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('uri', help='URI given by the server.')
    parser.add_argument('--mode', help="Mode to run in. "
                                       "live: interactive prompt to caption videos on the fly. "
                                       "headless: give list of video paths, caption, then exit.",
                        choices=['live', 'headless'],
                        default='live')
    parser.add_argument('--videos', nargs='+', help="Used only for headless mode. "
                                                    "Paths to video files separated by spaces.", required=False)
    parser.add_argument('--temp_dir', help="Temporary directory ", default='temp/')
    parser.add_argument('--feature_type', choices=available_features, default='nasnetalarge')
    parser.add_argument('--gpu_list', nargs='+', help="List of GPUs to use.", default=[1])

    args = parser.parse_args()

    _validate(args)
    watch(args)

