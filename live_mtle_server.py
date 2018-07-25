"""
Python2 script
"""

import argparse
import logging
import Pyro4
import numpy as np
import theano
import sys
sys.path.insert(1,'jobman')
sys.path.insert(1,'coco-caption')

from live_mtle_model_loader import LiveCaptioner

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def serpent_to_np(serpent_array):
    """
    Convert serpent basic array to np ndarray
    :param serpent_array:
    :return:
    """
    features_np = np.ndarray((len(serpent_array), len(serpent_array[0])))
    for i, feat in enumerate(serpent_array):
        features_np[i, :] = np.array(feat)
    
    features_np = features_np.astype(np.float32)
    return features_np


@Pyro4.expose
class Captioner(object):
    def __init__(self, model_checkpoint_dir):
        self.caption_service = LiveCaptioner(model_checkpoint_dir)

    def caption_features(self, recv_matrix):
        logger.debug("Recieved {} features".format(len(recv_matrix)))

        features_np = serpent_to_np(recv_matrix)
        caption = self.caption_service.caption(features_np)
        # print(features_np[0, :5])

        return caption


def listen(args):

    captioner_daemon = None
    try:
        captioner_daemon = Pyro4.Daemon()
        captioner = Captioner(args.model_checkpoint_dir)
        uri = captioner_daemon.register(captioner)

        print("Listening for requests. Use URI below to start client.")
        print("Captioner URI: {}".format(uri))

        captioner_daemon.requestLoop()
    except KeyboardInterrupt as e:
        print("Manual exit.")
        if captioner_daemon:
            captioner_daemon.close()
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint_dir', help="Directory of model checkpoint")
    args = parser.parse_args()

    listen(args)
