"""
Python3 script
"""

import os
import argparse
import logging
import Pyro4
import mss
import cv2
import asyncio
import time
import numpy as np

from data.process_frames import process_vid
from data.py3_process_features import create_batches, process_batches, available_features, init_model
from threading import Thread
from multiprocessing import Queue, Array
from ctypes import c_char

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
    uri = args.uri
    remote_captioner = Pyro4.Proxy(uri)

    if not os.path.isdir(args.temp_dir):
        mkdirs_safe(args.temp_dir)

    # Where we store frames before we begin feature processing. TODO: Could be a tempfile.TemporaryDirectory instead
    root_frames_dir = os.path.join(args.temp_dir, 'frames')
    if not os.path.isdir(root_frames_dir):
        mkdirs_safe(root_frames_dir)

    try:
        if args.mode == 'headless':
            load_img, tf_img, feats_model = init_model(args.gpu_list, args.feature_type)
            modelling_refs = (load_img, tf_img, feats_model, remote_captioner)

            for vpath in args.videos:
                caption = vid_path_to_caption(root_frames_dir, vpath, modelling_refs)
                logger.info("Caption for [{}]: {}".format(vpath, caption))
        elif args.mode == 'prompt':
            load_img, tf_img, feats_model = init_model(args.gpu_list, args.feature_type)
            modelling_refs = (load_img, tf_img, feats_model, remote_captioner)

            option = ''
            while option.lower() != 'q':
                print("List of options:")
                print("q: quit, c: provide a video path, cc: provide video paths infinitely")
                option = input("-> ").lower().strip()
                if option == 'c':
                    vpath = input("Provide path to video: ").replace('\n', "").replace("'", "").replace('"', "")
                    vpath = vpath.strip()
                    caption = vid_path_to_caption(root_frames_dir, vpath, modelling_refs)
                    print("CAPTION: " + caption)
                if option == 'cc':
                    vpath = ''
                    while vpath != 'q':
                        vpath = input("Provide path to video (q to exit): ").replace('\n', "").replace("'", "").replace('"', "")
                        vpath = vpath.strip()

                        if vpath != 'q' and vpath != 'Q':
                            caption = vid_path_to_caption(root_frames_dir, vpath, modelling_refs)
                            print("CAPTION: " + caption)
        elif args.mode == 'live':
            load_img, tf_img, feats_model = init_model(args.gpu_list, args.feature_type, fromfile=False)
            modelling_refs = (load_img, tf_img, feats_model, remote_captioner)

            live_caption_region(modelling_refs)

    except Exception as e:
        logger.exception(e)


def vid_path_to_caption(root_frames_dir, vpath, modelling_refs):
    (load_img, tf_img, feats_model, remote_captioner) = modelling_refs

    vid_name = vpath.split('/')[-1]
    frames_dir = os.path.join(root_frames_dir, vid_name)

    if os.path.isdir(frames_dir) and len(os.listdir(frames_dir)) != 0:
        logger.warning("Frames already exist at {}".format(frames_dir))
    elif os.path.isfile(frames_dir):
        logger.error("The video is placed in the frames directory. Please move it!")
        return "ERROR"
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
        return "ERROR"

    feats = process_batches(batches, args.feature_type, args.gpu_list, feats_model)

    # print(feats[0][:5])

    logger.info("Extracted {} features.".format(len(feats)))

    logger.info("Captioning...")
    return remote_captioner.caption_features(np_to_serpent(feats))


MAX_STR_LEN = 120
current_caption = Array(c_char, range(MAX_STR_LEN))
# feature_buffer = []
feature_hyprbatch_size = 16
# frame_buffer = []
fps_lock = 29.97
seconds_buffer_length = 3
frame_buffer_len = int(np.math.ceil(fps_lock * seconds_buffer_length))
inter_frame_delay = 1. / float(fps_lock)

shared_buffer = Queue(frame_buffer_len)


def caption_display(modelling_refs):
    load_img, tf_img, feats_model, remote_captioner = modelling_refs
    global shared_buffer
    global current_caption

    while True:
        time.sleep(0.5)
        # s = shared_buffer.get()
        # print("I got {}".format(s))
        # print('fps: {0}'.format(1 / (time.time() - last_time)))
        if shared_buffer.full():
            frames = [shared_buffer.get() for _ in range(frame_buffer_len)]
            for f in frames:
                # Check for poison pill
                if f is None:
                    return

            feats = cv_frames_to_feats(frames, modelling_refs)

            logger.info("Captioning...")
            caption = remote_captioner.caption_features(np_to_serpent(feats))
            try:
                current_caption.value = caption.encode()
                logger.debug("Got caption: {}".format(caption))
            except Exception as e:
                logger.exception(e)


def live_caption_region(modelling_refs):

    print("Let's configure your bounding box. Select a region on your screen to caption.")
    time.sleep(0.5)
    monitor_number = 1
    with mss.mss() as ss:
        whole = ss.grab(ss.monitors[monitor_number])

    img = np.array(whole)
    monitor = cv2.selectROI(img)
    cv2.destroyAllWindows()

    monitor = {'top': monitor[1], 'left': monitor[0], 'width': monitor[2], 'height': monitor[3]}

    # monitor = None

    # t1 = Thread(target=monitor_display, args=(monitor,))
    t1 = Thread(target=caption_display, args=(modelling_refs,))

    # t1.start()
    t1.start()

    global shared_buffer
    global current_caption

    # UI stuff will happen on main thread
    # https://github.com/opencv/opencv/issues/8407

    font = cv2.FONT_HERSHEY_SIMPLEX
    fps = 0

    # initial caption
    current_caption.value = "Awaiting captioner...".encode()

    with mss.mss() as sct:
        while True:
            # print("Tick")

            last_time = time.time()
            # time.sleep(1)

            img = np.array(sct.grab(monitor))
            if not shared_buffer.full():
                pil_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                shared_buffer.put(pil_img)

            textsize = cv2.getTextSize(current_caption.value.decode(), font, 1, 2)[0]
            txt_x_coord = int((monitor['width'] - textsize[0]) / 2)
            y_planes = monitor['height'] / 8
            # Put in bottom eight of image
            txt_y_coord = int(monitor['height'] - y_planes)

            cv2.putText(img, current_caption.value.decode(), (txt_x_coord, txt_y_coord), font,
                        0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, "fps: {}".format(int(fps)), (2, 22), font,
                        0.4, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('OpenCV SCT', img)
            time.sleep(inter_frame_delay)

            fps = 1 / (time.time() - last_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                while not shared_buffer.full():
                    # Poison pill
                    time.sleep(inter_frame_delay)
                    shared_buffer.put(None)

                logger.info("Shut down display. Sent signal to captioner...")
                break

    t1.join()

    # while True:
    #     last_time = time.time()
    #
    #     img = np.array(sct.grab(monitor))
    #
    #     cv2.imshow('OpenCV SCT', img)
    #     time.sleep(inter_frame_delay)
    #
    #     # print('fps: {0}'.format(1 / (time.time() - last_time)))
    #
    #     if cv2.waitKey(25) & 0xFF == ord('q'):
    #         cv2.destroyAllWindows()
    #         break
    #
    #     if len(frame_buffer) < frame_buffer_len:
    #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #         frame_buffer.append(img)
    #     else:
    #         feats = cv_frames_to_feats(frame_buffer, modelling_refs)
    #         frame_buffer = []
    #         logger.info("Captioning...")
    #         # caption =
    #         # print("CAPTION: ", caption)

    # if len(frame_buffer) <= feature_hyprbatch_size:
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     frame_buffer.append(img)
    # if len(frame_buffer) == feature_hyprbatch_size:
    #     # Ready to process
    #     feats = cv_frames_to_feats(frame_buffer, modelling_refs)
    #     feature_buffer.extend(feats)
    #     frame_buffer = []
    #
    # if len(feature_buffer) == frame_buffer_len:
    #     logger.info("Captioning...")
    #     caption = remote_captioner.caption_features(np_to_serpent(feature_buffer))
    #
    #     print("CAPTION: ", caption)
    #     feature_buffer = []


def cv_frames_to_feats(cv_frames, modelling_refs):
    (load_img, tf_img, feats_model, remote_captioner) = modelling_refs

    logger.info("Exracting features...")

    try:
        batches = create_batches(cv_frames, load_img, tf_img, batch_size=8)
    except OSError as e:
        logger.exception(e)
        logger.warning("Corrupt image file. Skipping...")
        return "ERROR"

    feats = process_batches(batches, args.feature_type, args.gpu_list, feats_model)

    # print(feats[0][:5])

    logger.info("Extracted {} features.".format(len(feats)))

    return feats


def _validate(args):
    if args.mode == 'headless':
        if not args.videos:
            raise AssertionError("You must use the --videos option for headless mode.")

        for vpath in args.videos:
            if not os.path.exists(vpath):
                raise IOError("No video exists at {}".format(vpath))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('uri', help='URI given by the server.')
    parser.add_argument('--mode', help="Mode to run in. "
                                       "live: perform captioning of a region on your screen. "
                                       "prompt: interactive prompt to caption videos on the fly"
                                       "headless: give list of video paths, caption, then exit.",
                        choices=['live', 'headless', 'prompt'],
                        default='prompt')
    parser.add_argument('--videos', nargs='+', help="Used only for headless mode. "
                                                    "Paths to video files separated by spaces.", required=False)
    parser.add_argument('--temp_dir', help="Temporary directory ", default='temp/')
    parser.add_argument('--feature_type', choices=available_features, default='nasnetalarge')
    parser.add_argument('--gpu_list', nargs='+', help="List of GPUs to use.", default=[1])

    args = parser.parse_args()

    _validate(args)
    watch(args)

