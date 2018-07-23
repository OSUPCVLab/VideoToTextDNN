"""
Python3 script
"""

import os
import argparse
import logging
import multiprocessing
from multiprocessing.connection import Client

from live_mtle_utils import Pickle2Reducer
from data.process_frames import process_vid
from data.py3_process_features import create_batches, process_batches, available_features, init_model

# multiprocessing.context._default_context.reducer = Pickle2Reducer()

ctx = multiprocessing.get_context()
ctx.reducer = Pickle2Reducer()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def mkdirs_safe(dir):
    try:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    except OSError as e:
        logger.exception(e)


def caption(sock, features):
    msg = ('caption', features)
    sock.send(msg)
    caption = sock.recv()

    return caption


def watch(args):
    load_img, tf_img, feats_model = init_model(args.gpu_list, args.feature_type)
    address = (args.server_ip, args.server_port)
    try:
        socket = Client(address)
    except Exception as e:
        logger.exception(e)
        exit(1)

    if not os.path.isdir(args.temp_dir):
        mkdirs_safe(args.temp_dir)

    for vpath in args.videos:
        vid_name = vpath.split('/')[-1]
        frames_dir = os.path.join(args.temp_dir, vid_name)

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
            continue

        feats = process_batches(batches, args.feature_type, args.gpu_list, feats_model)

        logger.info("Extracted {} features.".format(len(feats)))

        logger.info("Captioning...")
        print(caption(socket, feats))


def _validate(args):
    for vpath in args.videos:
        if not os.path.exists(vpath):
            raise IOError("No video exists at {}".format(vpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos', nargs='+', help="Paths to video files separated by spaces.")
    parser.add_argument('--temp_dir', help="Temporary directory ", default='temp/')
    parser.add_argument('--feature_type', choices=available_features, default='nasnetalarge')
    parser.add_argument('--gpu_list', nargs='+', help="List of GPUs to use.", default=[0])
    parser.add_argument('--server_ip', help="IP/hostname of model server.", default="localhost")
    parser.add_argument('--server_port', help="Port of model server.", default=45999)

    args = parser.parse_args()

    _validate(args)
    watch(args)

