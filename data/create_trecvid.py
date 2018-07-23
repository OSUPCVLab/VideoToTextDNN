import os
import argparse
import nltk
import sys
import numpy as np

from util import *

SEED = 9


def get_annots_trecvid(vid_feat_files, id_to_A_B_cap_dict, unittest, splits):
    vids_train = []
    vids_val = []
    vids_test = []
    all_vids = {}
    annotations = {}

    print 'Retrieving annotations...'
    if unittest:
        print 'UNIT TEST: On'
        id_to_A_B_cap_dict = {i: id_to_A_B_cap_dict[i] for enum, i in enumerate(id_to_A_B_cap_dict) if enum < unittest}

    n = len(id_to_A_B_cap_dict)

    # We are going to create the valid and test datasets ourselves.
    train_split, valid_split, test_split = splits.split(',')

    n_as_float = float(n)

    num_train = int(n_as_float * float(train_split))
    num_valid = int(n_as_float * float(valid_split))
    num_test = int(n - (num_valid + num_train))
    assert n == num_train + num_valid + num_test

    count_train = 0
    count_valid = 0
    count_test = 0

    for vid_id in vid_feat_files:
        if vid_id not in id_to_A_B_cap_dict:
            continue

        for enum, cap in enumerate(id_to_A_B_cap_dict[vid_id]):
            if not all_vids.has_key(vid_id):
                all_vids[vid_id] = 1
            else:
                all_vids[vid_id] += 1

            ocaption = cap
            ocaption = ocaption.replace('\n', '')
            ocaption = ocaption.strip()

            udata = ocaption.decode("utf-8", "ignore")
            ocaption = udata.encode("ascii", "ignore")

            tokens = nltk.word_tokenize(ocaption.replace('.', ''))

            if len(tokens) == 0:
                continue

            tokenized = ' '.join(tokens)
            tokenized = tokenized.lower()

            if annotations.has_key(vid_id):
                annotations[vid_id].append({'tokenized': tokenized, 'image_id': vid_id, 'cap_id': str(enum), 'caption': ocaption})
            else:
                annotations[vid_id]= []
                annotations[vid_id].append({'tokenized': tokenized, 'image_id': vid_id, 'cap_id': str(enum), 'caption': ocaption})

        if count_train < num_train:
            vids_train.extend([vid_id + '_' + str(enum) for enum, i in enumerate(annotations[vid_id])])
            count_train += 1
        elif count_valid < num_valid:
            vids_val.extend([vid_id + '_' + str(enum) for enum, i in enumerate(annotations[vid_id])])
            count_valid += 1
        elif count_test < num_test:
            vids_test.extend([vid_id + '_' + str(enum) for enum, i in enumerate(annotations[vid_id])])
            count_test += 1

    np.random.shuffle(vids_train)
    np.random.shuffle(vids_val)
    np.random.shuffle(vids_test)

    return annotations, vids_train, vids_val, vids_test, all_vids


def get_features_from_dir(vid_ids, feats_dir, feat_type):
    feats = {}

    for i, vid_id in enumerate(vid_ids):
        feat_file_path = os.path.join(feats_dir, vid_id.split('vid')[-1])

        if feat_type == 'c3d':
            feats[vid_id] = load_c3d_feat(feat_file_path)
            print('features extracted successfuly: ' + feat_file_path)
        else:
            if os.path.exists(feat_file_path):
                feat = np.load(feat_file_path)
                feats[vid_id] = feat
                print('features extracted successfuly: ' + feat_file_path)
            else:
                print('No features found!: ' + feat_file_path)

        print str(i) + '/' + str(len(vid_ids))
    return feats


def build_ground_truth_dict(gt_dir):
    gt_map_file = open(os.path.join(gt_dir, 'vtt.gt'), 'r')
    gt_A_file = open(os.path.join(gt_dir, 'vines.textDescription.A.testingSet'), 'r')
    gt_B_file = open(os.path.join(gt_dir, 'vines.textDescription.B.testingSet'), 'r')

    gt_A_index_to_cap_dict = {}
    for line in gt_A_file:
        cap_id, cap = line.replace('\n', '').split('    ')
        gt_A_index_to_cap_dict[cap_id] = cap
    gt_B_index_to_cap_dict = {}
    for line in gt_B_file:
        cap_id, cap = line.replace('\n', '').split('    ')
        gt_B_index_to_cap_dict[cap_id] = cap
    id_to_A_B_cap_dict = {}
    for line in gt_map_file:
        vid_id, cap_id_A, cap_id_B = line.replace('\n', '').split(' ')
        # vidID -> (capA, capB)
        id_to_A_B_cap_dict['vid' + vid_id] = (gt_A_index_to_cap_dict[cap_id_A], gt_B_index_to_cap_dict[cap_id_B])

    return id_to_A_B_cap_dict


def trecvid(params):
    pkl_dir = params.pkl_dir
    feats_dir = params.feats_dir
    gt_dir = params.gt_dir
    unittest = params.test
    splits = params.splits
    feat_type = params.type
    protocol = params.protocol

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    train_path = os.path.join(pkl_dir, 'train.pkl')
    valid_path = os.path.join(pkl_dir, 'valid.pkl')
    test_path = os.path.join(pkl_dir, 'test.pkl')
    cap_path = os.path.join(pkl_dir, 'CAP.pkl')
    dict_path = os.path.join(pkl_dir, 'worddict.pkl')

    if protocol != '':
        filename = 'FEATS_{}_{}.pkl'.format(feat_type, protocol)
    else:
        filename = 'FEATS_{}.pkl'.format(feat_type)

    feats_path = os.path.join(pkl_dir, filename)

    id_to_A_B_cap_dict = build_ground_truth_dict(gt_dir)
    vid_feat_files = ['vid' + i for i in os.listdir(feats_dir)]

    annotations, vids_train, vids_val, vids_test, all_vids = get_annots_trecvid(vid_feat_files, id_to_A_B_cap_dict, unittest, splits)

    dump_pkl(vids_train, train_path)
    print('train.pkl created')
    dump_pkl(vids_val, valid_path)
    print('valid.pkl created')
    dump_pkl(vids_test, test_path)
    print('test.pkl created')

    dump_pkl(all_vids.keys(), os.path.join(pkl_dir, 'allvids.pkl'))
    dump_pkl(annotations, cap_path)
    print('CAP.pkl created')

    worddict = create_dictionary(annotations, dict_path)
    dump_pkl(worddict, dict_path)
    print('worddict.pkl created')

    features = get_features_from_dir(annotations.keys(), feats_dir, feat_type)
    dump_pkl(features, feats_path)

    print 'FEAT file created! Path: {}'.format(feats_path)

    if params.do_skip_thoughts:
        logger.info("Generating skip-thoughts...")
        import create_skip_vectors
        class ArgsFaker():
            captions_file = cap_path
            output_file = os.path.join(pkl_dir, 'skip_vectors.pkl')

        fake_args = ArgsFaker()
        create_skip_vectors.main(fake_args)


if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-s', '--seed', type=int, help="Random seed.", default=SEED, required=False)
    arg_parser.add_argument('-f','--feats_dir',dest ='feats_dir',type=str,default='')
    arg_parser.add_argument('-gt','--gt_dir',dest ='gt_dir',type=str,default='')
    arg_parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,default='')
    arg_parser.add_argument('-type','--type',dest ='type',type=str,default='googlenet')
    arg_parser.add_argument('-t','--test',dest = 'test',type=int,default=0,
                            help='perform small unit test. If value 0 not unit test if greater than 0 gets a dataset with that numbers of videos')
    arg_parser.add_argument('-sp', '--splits', dest='splits', type=str, default='0.61,0.05,0.34',
                            help='Create validation and test datasets. Usage: floats delimited by commas, '
                                 'of the form Tr,Val. ex: {-s 0.60,0.20,0.20}. Default: 0.61,0.05,0.34')
    arg_parser.add_argument('-proc', '--protocol', dest='protocol', type=str, default='')
    arg_parser.add_argument('-st', '--do_skip_thoughts', dest='do_skip_thoughts', action='store_true', default=False)

    args = arg_parser.parse_args()

    np.random.seed(args.seed)

    if not len(sys.argv) > 1:
        print arg_parser.print_help()
        sys.exit(0)

    trecvid(args)
