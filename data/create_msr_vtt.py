import os
import numpy as np
import argparse
import json
import nltk
import logging

from util import *

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SEED = 9


def get_annots_vtt(feats_pool, train_val_list_path, test_list_path, test_sens_path, annotations, unittest=0):
    with open(train_val_list_path, 'r') as data_file:
        train_val_data = json.load(data_file)

    with open(test_list_path, 'r') as data_file:
        test_data = json.load(data_file)

    test_sens = None
    if test_sens_path:
        with open(test_sens_path, 'r') as data_file:
            test_sens = json.load(data_file)

    annotations, vids_train, vids_val, all_vids = get_annots_train_val_vtt(feats_pool, train_val_data, annotations, {}, unittest)
    annotations, vids_test, all_vids = get_annots_test_vtt(feats_pool, test_data, test_sens, annotations, all_vids, unittest)

    return annotations, vids_train, vids_val, vids_test, all_vids


def get_annots_train_val_vtt(feats_pool, train_val_data, annotations, all_vids, unittest=0):
    vids_train = []
    vids_val = []

    logger.info('Retrieving annotations for train-val...')

    videos_getting = [i['video_id'] for i in train_val_data['videos']]

    if unittest:
        num_videos = unittest
        logger.debug('UNIT TEST: On')
        np.random.shuffle(videos_getting)
        videos_getting = videos_getting[:num_videos]

    sentences = train_val_data['sentences']

    for sent in sentences:
        vid_name = sent['video_id']
        if vid_name not in videos_getting:
            continue

        if vid_name not in feats_pool:
            logger.warn("feature was missing for video_id={}".format(vid_name))
            continue

        if vid_name not in all_vids:
            all_vids[vid_name] = 1
        else:
            all_vids[vid_name] += 1

        ocaption = sent['caption']
        ocaption = ocaption.strip().encode('utf-8', 'ignore')

        tokens = nltk.word_tokenize(ocaption)
        tokenized = ' '.join(tokens)
        tokenized = tokenized.lower()

        if vid_name in annotations:
            cap_id = str(len(annotations[vid_name]))
            annotations[vid_name].append({'tokenized':tokenized,'image_id':vid_name,'cap_id':cap_id,'caption':ocaption})
        else:
            annotations[vid_name] = []
            cap_id = str(0)
            annotations[vid_name].append({'tokenized':tokenized,'image_id':vid_name,'cap_id':cap_id,'caption':ocaption})

        vid_and_cap = vid_name + '_' + cap_id

        vid_id = int(vid_name.split('video')[1])

        gt_id = train_val_data['videos'][vid_id]['id']
        assert gt_id == vid_id, 'Got an ID mis-match: vid_id={}, json_id={}'.format(vid_id, gt_id)

        if train_val_data['videos'][vid_id]['split'] == 'train':
            vids_train.append(vid_and_cap)
        elif train_val_data['videos'][vid_id]['split'] == 'validate':
            vids_val.append(vid_and_cap)
        else:
            raise ValueError("Video ID {} is not in train or valid split. Correct json file given?".format(vid_id))

    np.random.shuffle(vids_train)  # If we don't shuffle performance deminishes
    np.random.shuffle(vids_val)

    return annotations, vids_train, vids_val, all_vids


def get_annots_test_vtt(feats_pool, test_list_data, test_sens, annotations, all_vids, unittest=0):
    vids_test = []

    logger.info('Retrieving annotations for test...')

    videos_getting = [i['video_id'] for i in test_list_data['videos']]

    if unittest:
        num_videos = unittest
        logger.debug( 'UNIT TEST: On')
        np.random.shuffle(videos_getting)
        videos_getting = videos_getting[:num_videos]

    for vid_name in videos_getting:
        if vid_name not in videos_getting:
            continue

        if vid_name not in feats_pool:
            logger.warn("feature was missing for video_id={}".format(vid_name))
            continue

        if vid_name not in all_vids:
            all_vids[vid_name] = 1
        else:
            all_vids[vid_name] += 1

        if test_sens:
            # Use the released test sentences
            vid_sens = [s for s in test_sens['sentences'] if s['video_id'] == vid_name]

            for sent in vid_sens:
                ocaption = sent['caption']
                ocaption = ocaption.strip().encode('utf-8', 'ignore')

                tokens = nltk.word_tokenize(ocaption)
                tokenized = ' '.join(tokens)
                tokenized = tokenized.lower()

                if vid_name in annotations:
                    cap_id = str(len(annotations[vid_name]))
                    annotations[vid_name].append(
                        {'tokenized': tokenized, 'image_id': vid_name, 'cap_id': cap_id, 'caption': ocaption})
                else:
                    annotations[vid_name] = []
                    cap_id = str(0)
                    annotations[vid_name].append(
                        {'tokenized': tokenized, 'image_id': vid_name, 'cap_id': cap_id, 'caption': ocaption})

                vid_and_cap = vid_name + '_' + cap_id
                vids_test.append(vid_and_cap)

        else:
            ocaption = 'no caption'
            ocaption = ocaption.strip().encode('utf-8', 'ignore')

            tokens = nltk.word_tokenize(ocaption)
            tokenized = ' '.join(tokens)
            tokenized = tokenized.lower()

            annotations[vid_name] = []
            cap_id = str(0)
            annotations[vid_name].append(
                {'tokenized': tokenized, 'image_id': vid_name, 'cap_id': cap_id, 'caption': ocaption})

            vid_and_cap = vid_name + '_' + cap_id
            vids_test.append(vid_and_cap)

    np.random.shuffle(vids_test)

    return annotations, vids_test, all_vids


def load_annots_vtt(cap_path):
    return load_pkl(cap_path)


def get_features_from_dir(vid_frame_folder_names, feats_dir, feats_2017_test_dir, feat_type):

    feats = {}
    progress_checking = int(len(vid_frame_folder_names) / 10)

    logger.info("Extracting features...")

    for i, files in enumerate(vid_frame_folder_names):
        ext = '.' + files.split('.')[-1]
        feat_filename = files.split('/')[-1].split(ext)[0]

        vid_id = int(files.split('video')[1])
        if vid_id >= 10000:
            feat_file_path = os.path.join(feats_2017_test_dir, feat_filename)
        else:
            feat_file_path = os.path.join(feats_dir, feat_filename)

        if feat_type == 'c3d':
            feats[feat_filename]=load_c3d_feat(feat_file_path)
            logger.info('features extracted successfuly: ' + feat_file_path)
        else:
            if os.path.exists(feat_file_path):
                feat = np.load(feat_file_path)
                feats[feat_filename] = feat
                # print('features extracted successfuly: ' + feat_file_path)
            else:
                logger.info('No features found!: ' + feat_file_path)

        if i % progress_checking == 0:
            logger.info("Processed " + str(i) + '/' + str(len(vid_frame_folder_names)))

    return feats


def validate(vids_train, vids_val, vids_test):
    ntr = len(vids_train)
    logger.info("Have {} train samples".format(ntr))
    assert ntr > 0

    nva = len(vids_val)
    logger.info("Have {} val samples".format(nva))
    assert nva > 0

    nts = len(vids_test)
    logger.info("Have {} test samples.".format(nts))
    assert nts > 0

    tr_s = set(vids_train)
    va_s = set(vids_val)
    ts_s = set(vids_test)

    inter = tr_s.intersection(va_s)
    assert len(inter) == 0, 'Validation contaminated with training data.'
    inter = va_s.intersection(ts_s)
    assert len(inter) == 0, 'Testing contaminated with validation data.'
    inter = tr_s.intersection(ts_s)
    assert len(inter) == 0, 'Testing contaminated with training data.'


def vtt(params):
    pkl_dir = params.pkl_dir
    feats_dir = params.feats_dir
    feats_testing_dir = params.feats_testing_dir
    json_dir = params.json_dir
    unittest = params.test
    feat_type = params.type
    protocol = params.protocol
    version = params.version

    annotations = {}

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    train_path = os.path.join(pkl_dir,'train.pkl')
    valid_path = os.path.join(pkl_dir,'valid.pkl')
    test_path = os.path.join(pkl_dir,'test.pkl')
    cap_path = os.path.join(pkl_dir,'CAP.pkl')
    dict_path = os.path.join(pkl_dir,'worddict.pkl')

    if protocol != '':
        filename = 'FEATS_{}_{}.pkl'.format(feat_type, protocol)
    else:
        filename = 'FEATS_{}.pkl'.format(feat_type)

    feats_path = os.path.join(pkl_dir, filename)

    if os.path.exists(train_path) or os.path.exists(valid_path) or os.path.exists(test_path):
        var = raw_input("Pickle files found in [{}]. Do you want to erase them? type: [yes] [no] ".format(pkl_dir))

        if var == 'yes':
            logger.info('Removing old pkls...')
            remove_pickle_files(cap_path, dict_path, feats_path, test_path, train_path, valid_path)

        else:
            logger.info('Loading previous pickle files and creating new FEATS_ file at path: {}'.format(feats_path))
            if os.path.exists(feats_path):
                os.remove(feats_path)

            annotations = load_annots_vtt(cap_path)

            features = get_features_from_dir(annotations.keys(), feats_dir, feats_testing_dir, feat_type)
            dump_pkl(features, feats_path)
            logger.info('FEAT file created! Path: {}'.format(feats_path))
            sys.exit(0)

    vid_feats_dirs = os.listdir(feats_dir)
    vid_feats_dirs = sorted(vid_feats_dirs, key=lambda x: float(x.split('video')[-1])) #This is to sort the videos

    vid_testing_feats_dirs = os.listdir(feats_testing_dir)
    vid_testing_feats_dirs = sorted(vid_testing_feats_dirs, key=lambda x: float(x.split('video')[-1]))

    feats_pool = vid_feats_dirs + vid_testing_feats_dirs

    test_sens_path = None

    if version == '2016':
        test_list_path = os.path.join(json_dir, 'test_videodatainfo_nosen_2016.json')
        train_val_list_path = os.path.join(json_dir, 'train_val_videodatainfo.json')
        if args.with_sentences: test_sens_path = os.path.join(json_dir, 'videodatainfo_2017.json')
    else:
        test_list_path = os.path.join(json_dir, 'test_videodatainfo_nosen_2017.json')
        train_val_list_path = os.path.join(json_dir, 'videodatainfo_2017.json')
        if args.with_sentences: test_sens_path = os.path.join(json_dir, 'test_videodatainfo_2017.json')

    annotations, vids_train, vids_val, vids_test, all_vids = get_annots_vtt(feats_pool, train_val_list_path,
                                                                            test_list_path, test_sens_path, annotations, unittest)

    logger.info('Validating generated lists...')
    validate(vids_train, vids_val, vids_test)

    dump_pkl(vids_test, test_path)
    logger.info('test.pkl created')

    dump_pkl(vids_train,train_path)
    logger.info('train.pkl created')

    dump_pkl(vids_val,valid_path)
    logger.info('valid.pkl created')

    dump_pkl(all_vids.keys(), os.path.join(pkl_dir,'allvids.pkl'))
    dump_pkl(annotations, cap_path)
    logger.info('CAP.pkl created')
    worddict = create_dictionary(annotations,dict_path)
    dump_pkl(worddict,dict_path)
    logger.info('worddict.pkl created')

    features = get_features_from_dir(annotations.keys(), feats_dir, feats_testing_dir, feat_type)
    dump_pkl(features,feats_path)
    logger.info('FEAT file created! Path: {}'.format(feats_path))

    if params.do_skip_thoughts:
        logger.info("Generating skip-thoughts...")
        import create_skip_vectors
        class ArgsFaker():
            captions_file = cap_path
            output_file = os.path.join(pkl_dir, 'skip_vectors.pkl')

        fake_args = ArgsFaker()
        create_skip_vectors.main(fake_args)


def remove_pickle_files(cap_path, dict_path, feats_path, test_path, train_path, valid_path):
    if os.path.exists(train_path):
        os.remove(train_path)
    if os.path.exists(valid_path):
        os.remove(valid_path)
    if os.path.exists(test_path):
        os.remove(test_path)
    if os.path.exists(cap_path):
        os.remove(cap_path)
    if os.path.exists(dict_path):
        os.remove(dict_path)
    if os.path.exists(feats_path):
        os.remove(feats_path)
    # if os.path.exists('allvids.pkl'):
    #     os.remove('allvids.pkl')


def _validate(args):
    if args.version == '2016' and args.with_sentences:
        logger.info("2016 version test sentences were made available in 2017 dataset.")
        sens_path = os.path.join(args.json_dir, "videodatainfo_2017.json")
        if os.path.exists(sens_path):
           logger.info("Found ground truth captions for 2016 test sentences")
        else:
            logger.critical("Did not find ground truth captions for 2016 test sentences: {}".format(sens_path))
            sys.exit(1)

    if args.type not in args.feats_dir or args.type not in args.feats_testing_dir:
        logger.critical("Requested feature type {}, but directories are something else:\tfeats_dir={}\tfeats_testing_dir={}".format(args.type, args.feats_dir, args.feats_testing_dir))
        sys.exit(1)


if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()

    creation_args = arg_parser.add_argument_group("CreationArgs")
    creation_args.add_argument('-s', '--seed', type=int, help="Random seed.", default=SEED, required=False)
    creation_args.add_argument('-f','--feats_dir',dest ='feats_dir',type=str, required=True)
    creation_args.add_argument('-ft', '--feats_testing_dir', dest='feats_testing_dir', type=str, required=True)
    creation_args.add_argument('-j','--json_dir',dest ='json_dir',type=str, required=True)
    creation_args.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str, required=True)
    creation_args.add_argument('-type','--type',dest ='type',type=str, required=True)
    creation_args.add_argument('-t', '--test', dest='test', type=int, default=0,
                               help='perform small unit test. If value 0 not unit test if greater than 0 gets a dataset with that numbers of videos')
    creation_args.add_argument('-proc', '--protocol', dest='protocol', type=str, default='')
    creation_args.add_argument('-st', '--do_skip_thoughts', dest='do_skip_thoughts', action='store_true', default=False)

    vtt_args = arg_parser.add_argument_group("VTTArgs")
    vtt_args.add_argument('-v', '--version', dest='version', type=str, default='2016', help="Which MSR-VTT version to create.", choices=['2016', '2017'])
    vtt_args.add_argument('-ws', '--with_sentences', dest='with_sentences', default=False, action='store_true', help='Use the available test set sentences.')

    args = arg_parser.parse_args()

    np.random.seed(args.seed)

    if not len(sys.argv) > 1:
        print(arg_parser.print_help())
        sys.exit(0)

    _validate(args)

    vtt(args)



