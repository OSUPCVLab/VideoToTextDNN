__author__ = 'oliver'

import argparse
import nltk
import shutil
import numpy as np
import logging
from util import *

import collections
from collections import OrderedDict

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEED = 9


def get_annots_lsmdc(filename, annotations, num_test):
    vids_names = {}

    with open(filename) as csvfile:
        rows = csvfile.readlines()
        for row in rows[:num_test]:
            row = row.split('\t')
            vid_name = row[0]

            if len(row) > 5:
                ocaption = row[5]

                ocaption = ocaption.replace('\n', '')
                udata = ocaption.decode("utf-8")
                caption = udata.encode("ascii", "ignore")

                tokens = nltk.word_tokenize(caption)
                tokenized = ' '.join(tokens)
                tokenized = tokenized.lower()

                if vids_names.has_key(vid_name):
                    vids_names[vid_name] += 1
                    logger.info('other annots')
                else:
                    feat_path = '/PATH/TO/lsmdc16/videos/' + vid_name + '.avi'
                    if not os.path.exists(feat_path):
                        logger.warning('video not found ' + feat_path)
                    vids_names[vid_name] = 1

                annotations[vid_name] = [
                    {'tokenized': tokenized, 'image_id': vid_name, 'cap_id': vids_names[vid_name], 'caption': ocaption}]

    return annotations, vids_names


def get_blind_lsmdc(filename, num_test):
    vids_names = OrderedDict()
    # annotations = OrderedDict()

    with open(filename) as csvfile:
        rows = csvfile.readlines()
        for row in rows:
            row = row.split('\t')
            vid_name = row[0]

            if vids_names.has_key(vid_name):
                vids_names[vid_name] += 1
                logger.info('other annots')
            else:
                # feat_path = '/media/onina/sea2/datasets/lsmdc/features_chal/'+vid_name
                # if not os.path.exists(feat_path):
                #     print 'features not found '+feat_path
                vids_names[vid_name] = 1

                # annotations[vid_name]=[{'tokenized':tokenized,'image_id':vid_name,'cap_id':1,'caption':''}]

    return vids_names


def get_annots_mvad(rows, annots_corpus, annotations, feats_dir):
    vids_names = {}

    for i, row in enumerate(rows):

        # row = row.split('\t')
        vid_name = row.split('/')[-1].split('.')[0]
        caption = annots_corpus[i]
        caption = caption.replace('\n', '')

        udata = caption.decode("utf-8")
        caption = udata.encode("ascii", "ignore")

        tokens = nltk.word_tokenize(caption)
        tokenized = ' '.join(tokens)
        tokenized = tokenized.lower()

        if vids_names.has_key(vid_name):
            vids_names[vid_name] += 1
            logger.info('other annots, there should be only 1')
            # sys.exit(0)
        else:
            vids_names[vid_name] = 1

        annotations[vid_name] = [
            {'tokenized': tokenized, 'image_id': vid_name, 'cap_id': vids_names[vid_name], 'caption': caption}]

    return annotations, vids_names


def create_dictionary(annotations, pkl_dir):
    worddict = collections.OrderedDict()
    word_idx = 2
    for a in annotations:
        caps = annotations[a]

        for cap in caps:
            tokens = cap['tokenized'].split()
            for token in tokens:
                if token not in ['', '\t', '\n', ' ']:
                    if not worddict.has_key(token):
                        worddict[token] = word_idx
                        word_idx += 1

    return worddict


def lsmdc16(args):
    data_dir = args.data_dir
    pkl_dir = args.pkl_dir

    num_train, num_valid, num_test, num_blind = 9999999999, 9999999999, 9999999999, 9999999999

    test_mode = int(args.unit_test)

    train_list_path = 'LSMDC16_annos_training.csv'
    valid_list_path = 'LSMDC16_annos_val.csv'
    test_list_path = 'LSMDC16_annos_test.csv'
    btest_list_path = 'LSMDC16_annos_blindtest.csv'

    if test_mode:
        num_train = int(0.50 * test_mode)
        num_test = int(0.15 * test_mode)
        num_valid = int(0.25 * test_mode)
        num_blind = test_mode - (num_test + num_train + num_valid)

    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []

    train_path = os.path.join(pkl_dir, 'train.pkl')
    if not os.path.exists(train_path):
        train_file = os.path.join(data_dir, train_list_path)
        logger.info(train_file)
        annotations, vids_names = get_annots_lsmdc(train_file, annotations, num_train)
        training_list = vids_names.keys()
        dump_pkl(training_list, train_path)
    else:
        training_list = load_pkl(train_path)

    all_vids = all_vids + training_list

    valid_path = os.path.join(pkl_dir, 'valid.pkl')
    if not os.path.exists(valid_path):
        valid_file = os.path.join(data_dir, valid_list_path)
        annotations, vids_names = get_annots_lsmdc(valid_file, annotations, num_valid)
        valid_list = vids_names.keys()
        dump_pkl(valid_list, valid_path)
    else:
        valid_list = load_pkl(valid_path)

    all_vids = all_vids + valid_list

    test_path = os.path.join(pkl_dir, 'test.pkl')
    if not os.path.exists(test_path):
        test_file = os.path.join(data_dir, test_list_path)
        annotations, vids_names = get_annots_lsmdc(test_file, annotations, num_test)
        test_list = vids_names.keys()
        dump_pkl(test_list, test_path)
    else:
        test_list = load_pkl(test_path)

    all_vids = all_vids + test_list

    cap_path = os.path.join(pkl_dir, 'CAP.pkl')
    if not os.path.exists(cap_path):
        dump_pkl(annotations, cap_path)

    dict_path = os.path.join(pkl_dir, 'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations, dict_path)
        dump_pkl(worddict, dict_path)

    btest_path = os.path.join(pkl_dir, 'blindtest.pkl')
    if not os.path.exists(btest_path):
        btest_file = os.path.join(data_dir, btest_list_path)
        vids_names = get_blind_lsmdc(btest_file, num_blind)
        btest_list = vids_names.keys()
        dump_pkl(btest_list, btest_path)
    else:
        btest_list = load_pkl(btest_path)

    logger.info('done creating dataset')


def mpii(params):
    data_dir = params.data_dir
    pkl_dir = params.pkl_dir
    testing = params.unit_test
    local_dir = params.local_dir
    feats_dir = params.feats_dir

    f = open(os.path.join(data_dir, 'lists', 'downloadLinksAvi.txt'), 'rb')
    files = f.readlines()
    f.close()
    f = open(os.path.join(data_dir, 'lists', 'annotations-someone.csv'), 'rb')
    annots = f.readlines()
    f.close()
    f = open(os.path.join(data_dir, 'lists', 'dataSplit.txt'), 'rb')
    splits_file = f.readlines()
    splits = {}

    annotations = {}
    train_clip_names = []
    valid_clip_names = []
    test_clip_names = []

    if testing:
        tuples = [(f, a) for f, a in zip(files, annots)]
        np.random.shuffle(tuples)
        tuples = tuples[:testing]
        files = [a[0] for a in tuples]
        annots = [b[1] for b in tuples]

    train_path = os.path.join(pkl_dir, 'train.pkl')
    if not os.path.exists(train_path):
        for line in splits_file:
            film_name = line.split('\t')[0]
            split = line.split('\t')[1]
            splits[film_name] = split.replace('\r\n', '')

        for i, file in enumerate(files):
            parts = file.split('/')

            film_name = parts[6]
            clip_name = parts[7].replace('\n', '')
            clip_name = clip_name.split('.avi')[0]
            caption = annots[i].split('\t')[1]
            caption = caption.replace('\n', '')

            udata = caption.decode("utf-8")
            caption = udata.encode("ascii", "ignore")

            tokens = nltk.word_tokenize(caption)
            tokenized = ' '.join(tokens)
            tokenized = tokenized.lower()

            annotations[clip_name] = [{'tokenized': tokenized, 'image_id': clip_name, 'cap_id': 1, 'caption': caption}]

            if splits[film_name] == 'training':
                train_clip_names.append(clip_name)
            elif splits[film_name] == 'validation':
                valid_clip_names.append(clip_name)
            elif splits[film_name] == 'test':
                test_clip_names.append(clip_name)

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []

    train_path = os.path.join(pkl_dir, 'train.pkl')
    if not os.path.exists(train_path):
        dump_pkl(train_clip_names, train_path)
    else:
        train_clip_names = load_pkl(train_path)

    all_vids = all_vids + train_clip_names

    valid_path = os.path.join(pkl_dir, 'valid.pkl')
    if not os.path.exists(valid_path):
        dump_pkl(valid_clip_names, valid_path)
    else:
        valid_clip_names = load_pkl(valid_path)

    all_vids = all_vids + valid_clip_names

    test_path = os.path.join(pkl_dir, 'test.pkl')
    if not os.path.exists(test_path):
        dump_pkl(test_clip_names, test_path)
    else:
        test_clip_names = load_pkl(test_path)

    all_vids = all_vids + test_clip_names

    cap_path = os.path.join(pkl_dir, 'CAP.pkl')
    if not os.path.exists(cap_path):
        dump_pkl(annotations, cap_path)

    dict_path = os.path.join(pkl_dir, 'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations, dict_path)
        dump_pkl(worddict, dict_path)

    if testing and local_dir:
        logger.info("Copying required features...")
        if not os.path.isdir(local_dir):
            os.makedirs(local_dir)

        for vid_name in all_vids:
            ft_path = os.path.join(feats_dir, vid_name)
            local_ft_path = os.path.join(local_dir, vid_name)
            shutil.copy2(ft_path, local_ft_path)

    logger.info('done creating dataset')


def mvad(params):
    feats_dir = params.feats_dir
    data_dir = params.data_dir
    pkl_dir = params.pkl_dir

    testing = params.unit_test
    local_dir = params.local_dir

    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    all_vids = []

    s_paths = [os.path.join(pkl_dir, 'train.pkl'),
               os.path.join(pkl_dir, 'valid.pkl'),
               os.path.join(pkl_dir, 'test.pkl')
               ]
    l_paths = [os.path.join(data_dir, 'lists/TrainList.txt'),
               os.path.join(data_dir, 'lists/ValidList.txt'),
               os.path.join(data_dir, 'lists/TestList.txt')
               ]
    c_paths = [os.path.join(data_dir, 'lists/TrainCorpus.txt'),
               os.path.join(data_dir, 'lists/ValidCorpus.txt'),
               os.path.join(data_dir, 'lists/TestCorpus.txt')
               ]

    for i, s_path in enumerate(s_paths):
        if not os.path.exists(s_path):
            _rows = open(l_paths[i], 'rw').readlines()
            _corpus = open(c_paths[i], 'rw').readlines()

            if testing:
                _pairs = [(r, c) for r, c in zip(_rows, _corpus)]
                np.random.shuffle(_pairs)
                num = int(testing * params.split[i])
                _rows = [p[0] for p in _pairs[:num]]
                _corpus = [p[1] for p in _pairs[:num]]

            annotations, vids_names = get_annots_mvad(_rows, _corpus, annotations, feats_dir)
            _list = vids_names.keys()
            dump_pkl(_list, s_path)
        else:
            _list = load_pkl(s_path)

        all_vids = all_vids + _list

    cap_path = os.path.join(pkl_dir, 'CAP.pkl')
    if not os.path.exists(cap_path):
        dump_pkl(annotations, cap_path)

    dict_path = os.path.join(pkl_dir, 'worddict.pkl')
    if not os.path.exists(dict_path):
        worddict = create_dictionary(annotations, dict_path)
        dump_pkl(worddict, dict_path)

    if testing and local_dir:
        logger.info("Copying required features...")
        if not os.path.isdir(local_dir):
            os.makedirs(local_dir)

        for vid_name in all_vids:
            ft_path = os.path.join(feats_dir, vid_name)
            local_ft_path = os.path.join(local_dir, vid_name)
            shutil.copy2(ft_path, local_ft_path)

    logger.info('done creating dataset')


def get_human_annotations(data_dir):
    hannot_path = os.path.join(data_dir, 'human_annotations', 'HumanCaps.csv')
    import csv

    hannot = {}
    with open(hannot_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='\"')
        for row in spamreader:
            logger.info(', '.join(row))
            hannot[row[0]] = row[1]
    return hannot


def tokenize_cap(caption):
    udata = caption.decode("utf-8")
    caption = udata.encode("ascii", "ignore")

    tokens = nltk.word_tokenize(caption)
    tokenized = ' '.join(tokens)
    tokenized = tokenized.lower()
    return tokenized


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    creation_args = parser.add_argument_group("CreationArgs")
    creation_args.add_argument('-s', '--seed', type=int, help="Random seed.", default=SEED, required=False)
    creation_args.add_argument('-d', '--data_dir', dest='data_dir', help='Example: /path/to/dataset/annotations',
                               required=True)
    creation_args.add_argument('-p', '--pkl_dir', dest='pkl_dir', help='Example: /path/to/dataset/pkls', required=True)
    creation_args.add_argument('-dbname', '--dbname', dest='dbname', help='Dataset type.', required=True,
                               choices=['mvad', 'mpii', 'lsmdc16'])
    creation_args.add_argument('-st', '--do_skip_thoughts', dest='do_skip_thoughts', action='store_true', default=False)

    ut_args = parser.add_argument_group("UnitTestArgs")
    ut_args.add_argument('-t', '--unit_test', dest='unit_test', type=int, default=0,
                         help='Perform small test. Takes number of samples in unit test dataset.')
    ut_args.add_argument('-l', '--local_dir', dest='local_dir', help="Where to copy unit_test features.", default=None)
    ut_args.add_argument('-sp', '--split', dest='split', nargs='+',
                         help='Space delimited [train val test] Data split to use in unit test dataset',
                         default=[0.50, 0.25, 0.25], type=float)
    ut_args.add_argument('-feat', '--feats_dir', dest='feats_dir', help='Example: /path/to/dataset/features_googlenet',
                         required=False)

    args = parser.parse_args()

    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit(0)

    np.random.seed(args.seed)

    if not args.feats_dir:
        if args.local_dir:
            logger.critical(
                "You must provide an argument for --feats_dir to create a local copy of features (--local_dir)")
            sys.exit(1)

    if args.dbname == 'mvad':
        mvad(args)
    if args.dbname == 'mpii':
        mpii(args)
    if args.dbname == 'lsmdc16':
        lsmdc16(args)
