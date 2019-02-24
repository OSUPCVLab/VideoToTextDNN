"""
Python2 script
"""

import argparse
import logging
import Pyro4
import numpy as np
import sys

from live_mtle_model_loader import MTLECaptioner, LSTMDDCaptioner, BaselineCaptioner

sys.path.insert(1,'jobman')
sys.path.insert(1,'coco-caption')


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
    def __init__(self, args):
        if args.model_type == 'mtle':
            self.caption_service = MTLECaptioner(args.model_checkpoint_dir)
        elif args.model_type == 'lstmdd':
            self.caption_service = LSTMDDCaptioner(args.model_checkpoint_dir)
        elif args.model_type == 'baseline':
            self.caption_service = BaselineCaptioner(args.model_checkpoint_dir)
        else:
            raise ValueError("Invalid model type specified:\t{}".format(args.model_type))

    def caption_features(self, recv_matrix):
        logger.debug("Recieved {} features".format(len(recv_matrix)))

        features_np = serpent_to_np(recv_matrix)
        caption = self.caption_service.caption(features_np)

        return caption


def listen(args):

    captioner_daemon = None
    try:
        captioner_daemon = Pyro4.Daemon()
        captioner = Captioner(args)
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
    model_choices = ['mtle', 'lstmdd', 'baseline']
    parser.add_argument('model_checkpoint_dir', help="Directory of model checkpoint")
    parser.add_argument('--model_type',
                        help="Model architecture to use. choices={}".format(model_choices),
                        choices=model_choices,
                        default='mtle', required=False)
    args = parser.parse_args()

    listen(args)
