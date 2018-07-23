import cPickle
import os
import numpy as np
import sys
import logging

from collections import OrderedDict


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def dump_pkl(obj, path):
    """
    Save a Python object into a pickle file.
    """
    f = open(path, 'wb')
    try:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)

    finally:
        f.close()
    print path+' created'


def load_pkl(path):
    """
    Load a pickled file.

    :param path: Path to the pickled file.

    :return: The unpickled Python object.
    """
    f = open(path, 'rb')
    try:
        rval = cPickle.load(f)
    finally:
        f.close()
    return rval

def create_dictionary(annotations,pkl_dir):
    worddict = OrderedDict()
    word_idx = 2
    for a in annotations:
        caps = annotations[a]

        for cap in caps:
            tokens = cap['tokenized'].split()
            for token in tokens:
                if token not in ['','\t','\n',' ']:
                    if not worddict.has_key(token):
                        worddict[token]=word_idx
                        word_idx+=1

    return worddict


def pad_frames(frames, limit):
    last_frame = frames[-1]
    padding = np.asarray([last_frame * 0.]*(limit-len(frames)))
    frames_padded = np.concatenate([frames, padding], axis=0)
    return frames_padded


def extract_frames_equally_spaced(frames, K):
    # chunk frames into 'how_many' segments and use the first frame
    # from each segment
    n_frames = len(frames)
    splits = np.array_split(range(n_frames), K)
    idx_taken = [s[0] for s in splits]
    sub_frames = frames[idx_taken]
    return sub_frames


def get_sub_frames(frames):

    K=28
    if len(frames) < K:
        frames_ = pad_frames(frames, K)
    else:
        frames_ = extract_frames_equally_spaced(frames, K)

    return frames_


def load_c3d_feat(feat_file_path):
    if os.path.exists(feat_file_path):
        files = os.listdir(feat_file_path)
        files.sort()
        allftrs = np.zeros((len(files), 4101),dtype=np.float32)

        for j in range(0, len(files)):
            feat = np.fromfile(os.path.join(feat_file_path, files[j]),dtype=np.float32)
            allftrs[j,:] = feat
        allftrs = get_sub_frames(allftrs)

        return allftrs
    else:
        print 'error feature file doesnt exist'+feat_file_path
        sys.exit(0)


def mkdirs_safe(dir):
    try:
        if not os.path.isdir(dir):
            os.makedirs(dir)
    except OSError as e:
        logger.exception(e)


def create_line(seed, dataset, annots_dir, feature_type, pickle_dir, feature_dir, feature_test_dir, ut=0, st=False):
    if dataset == 'mvad' or dataset == 'mpii' or dataset == 'lsmdc16':
        line = "python create_mvad_mpii_lsmdc.py "
        line += "-s {} ".format(seed)
        line += "-d {} ".format(annots_dir)
        line += "-p {} ".format(pickle_dir)
        line += "-dbname {} ".format(dataset)
    elif dataset == 'tacos':
        line = "python create_tacos.py "
        line += "-s {} ".format(seed)
        line += "-f {} ".format(feature_dir)
        line += "-gt {} ".format(annots_dir)
        line += "-p {} ".format(pickle_dir)
    elif dataset == 'youtube2text':
        line = "python create_y2t.py "
        line += "-s {} ".format(seed)
        line += "-f {} ".format(feature_dir)
        line += "-j {} ".format(annots_dir)
        line += "-p {} ".format(pickle_dir)
        line += "-type {} ".format(feature_type)
    elif dataset == 'vtt16':
        line = "python create_msr_vtt.py "
        line += "-s {} ".format(seed)
        line += "-f {} ".format(feature_dir)
        line += "-ft {} ".format(feature_test_dir)
        line += "-j {} ".format(annots_dir)
        line += "-p {} ".format(pickle_dir)
        line += "-type {} ".format(feature_type)
        line += "-v 2016 "
        line += "-ws "
    elif dataset == 'vtt17':
        line = "python create_msr_vtt.py "
        line += "-s {} ".format(seed)
        line += "-f {} ".format(feature_dir)
        line += "-ft {} ".format(feature_test_dir)
        line += "-j {} ".format(annots_dir)
        line += "-p {} ".format(pickle_dir)
        line += "-type {} ".format(feature_type)
        line += "-v 2017 "
        line += "-ws "
    elif dataset == 'trecvid':
        line = "python create_trecvid.py "
        line += "-s {} ".format(seed)
        line += "-f {} ".format(feature_dir)
        line += "-gt {} ".format(annots_dir)
        line += "-p {} ".format(pickle_dir)
        line += "-type {} ".format(feature_type)
    else:
        raise NotImplementedError("Dataset not implemented: {}".format(dataset))

    if ut:
        line += "-t {} ".format(ut)
    if st:
        line += "-st "

    return line
