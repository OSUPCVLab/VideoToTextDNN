import os
import argparse
import nltk
import cPickle
import sys
import numpy as np

from util import *
import create_msr_vtt


SEED = 9

def get_features_from_dir(vid_frame_folder_names, feats_dir, feat_type):
    feats = {}

    for i, files in enumerate(vid_frame_folder_names):
        ext = '.' + files.split('.')[-1]
        feat_filename = files.split('/')[-1].split(ext)[0]

        feat_file_path = os.path.join(feats_dir, feat_filename)

        if feat_type == 'c3d':
            feats[feat_filename] = load_c3d_feat(feat_file_path)
            print('features extracted successfuly: ' + feat_file_path)
        else:
            if os.path.exists(feat_file_path):
                feat = np.load(feat_file_path)
                feats[feat_filename] = feat
                print('features extracted successfuly: ' + feat_file_path)
            else:
                print('No features found!: ' + feat_file_path)

        print str(i) + '/' + str(len(vid_frame_folder_names))
    return feats


def get_annots_y2t(vid_caption_dict, youtube_map_dict, unittest=0, splits=''):
    vids_train = []
    vids_val = []
    vids_test = []
    all_vids = {}
    annotations = {}

    print 'Retrieving annotations...'

    pkl = youtube_map_dict
    if unittest:
        print 'UNIT TEST: On'
        keys = pkl.keys()
        np.random.shuffle(keys)
        keys = keys[:unittest]
        pkl = {key: pkl[key] for key in keys}

    n = len(pkl)

    if splits == 'yao':
        num_train = 1201
        num_valid = 100
        num_test = 670
    else:
        train_split, valid_split, test_split = splits.split(',')

        n_as_float = float(n)

        num_train = int(n_as_float * float(train_split))
        num_valid = int(n_as_float * float(valid_split))
        num_test = int(n_as_float * float(test_split))
        assert n == num_train + num_valid + num_test

    count_train = 0
    count_valid = 0
    count_test = 0

    for vid_name in pkl.keys():
        vid = youtube_map_dict[vid_name]

        for cap_id, cap in enumerate(vid_caption_dict[vid_name]):
            if not all_vids.has_key(vid_name):
                all_vids[vid_name] = 1
            else:
                all_vids[vid_name] += 1

            ocaption = cap
            ocaption = ocaption.replace('\n', '')
            ocaption = ocaption.strip()

            udata = ocaption.decode("utf-8")
            ocaption = udata.encode("ascii", "ignore")

            tokens = nltk.word_tokenize(ocaption.replace('.', ''))

            if len(tokens) == 0:
                continue

            tokenized = ' '.join(tokens)
            tokenized = tokenized.lower()

            if annotations.has_key(vid):
                annotations[vid].append({'tokenized': tokenized, 'image_id': vid, 'cap_id': str(cap_id), 'caption': ocaption})
            else:
                annotations[vid]= []
                annotations[vid].append({'tokenized': tokenized, 'image_id': vid, 'cap_id': str(cap_id), 'caption': ocaption})

        if count_train < num_train:
            vids_train.extend([vid + '_' + str(enum) for enum, i in enumerate(annotations[vid])])
            count_train += 1
        elif count_valid < num_valid:
            vids_val.extend([vid + '_' + str(enum) for enum, i in enumerate(annotations[vid])])
            count_valid += 1
        elif count_test < num_test:
            vids_test.extend([vid + '_' + str(enum) for enum, i in enumerate(annotations[vid])])
            count_test += 1

    np.random.shuffle(vids_train)
    np.random.shuffle(vids_val)
    np.random.shuffle(vids_test)

    return annotations, vids_train, vids_val, vids_test, all_vids


def get_features_from_pkl(from_pkl_file, all_vids_dict, youtube_map_dict):
    pkl = cPickle.load(open(from_pkl_file))
    feats = {}

    for key in all_vids_dict:
        # key is going to be of the form xxxxxxxxxx_##_## but we want vid####
        vid = youtube_map_dict[key]
        feats[vid] = pkl[vid]

    return feats


def fix_feature_file_names(youtube_map_dict, feats_dir, pkl_dir):
    feat_files = os.listdir(feats_dir)
    work_order = []
    for original in feat_files:
        if original not in youtube_map_dict.values():
            new_name = youtube_map_dict[original]
            did = "{} to {}".format(original, new_name)
            work_order.append(did)
            #print did
            orig_path = os.path.join(feats_dir, original)
            new_path = os.path.join(feats_dir, new_name)
            os.rename(orig_path, new_path)

    # Print to file a record of what names were changed
    work_order_path = os.path.join(pkl_dir, 'feat_name_changes.txt')
    f = open(work_order_path, 'w')
    for i in work_order:
        f.write(i + '\n')

    print "Saved name changes to {}".format(work_order_path)


def y2t(params):
    pkl_dir = params.pkl_dir
    feats_dir = params.feats_dir
    json_dir = params.json_dir
    unittest = params.test
    splits = 'yao' if params.yao else params.splits
    feat_type = params.type
    protocol = params.protocol
    from_pkl = params.from_pkl

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    if splits == 'yao':
        print("Using Yao2015 splits.")

    f = open(os.path.join(json_dir, 'dict_movieID_caption.pkl'), 'r')
    vid_caption_dict = cPickle.load(f)

    f = open(os.path.join(json_dir, 'dict_youtube_mapping.pkl'), 'r')
    youtube_map_dict = cPickle.load(f)

    if os.path.isdir(feats_dir):
        feat_files = set(os.listdir(feats_dir))
        vidX_formatted_files = set(youtube_map_dict.values())

        diff = feat_files - vidX_formatted_files
        if len(diff) > 0 and not from_pkl:
            print "Found mismatch of feature file names and youtube_mapping_dict." \
                  "Feature files will be re-named according to youtube_map_dict.pkl"
            fix_feature_file_names(youtube_map_dict, feats_dir, pkl_dir)

    else:
        print "Feature directroy not found at {}.\nExiting.".format(feats_dir)
        sys.exit(0)

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

    if os.path.exists(train_path) or os.path.exists(valid_path) or os.path.exists(test_path):
        var = raw_input("Pickle files found in [{}]. Do you want to erase them? type: yes/[no] ".format(pkl_dir))

        if var == 'yes':
            print 'Removing old pkls...'
            create_msr_vtt.remove_pickle_files(cap_path, dict_path, feats_path, test_path, train_path, valid_path)

        else:
            print('Loading previous pickle files and creating new FEATS_ file at path: {}'.format(feats_path))
            if os.path.exists(feats_path):
                os.remove(feats_path)

            annotations = create_msr_vtt.load_annots_vtt(cap_path)

            features = get_features_from_dir(annotations.keys(), feats_dir, feat_type)
            create_msr_vtt.dump_pkl(features, feats_path)
            print 'FEAT file created! Path: {}'.format(feats_path)
            sys.exit(0)

    annotations, vids_train, vids_val, vids_test, all_vids = get_annots_y2t(vid_caption_dict, youtube_map_dict,
                                                                            unittest, splits)

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

    if from_pkl:
        # Getting features from pkl file.
        from_pkl_file = os.path.join(feats_dir, 'FEAT_key_vidID_value_features.pkl')
        print "Loading features from pkl file."
        features = get_features_from_pkl(from_pkl_file, all_vids, youtube_map_dict)
    else:
        features = get_features_from_dir(annotations.keys(), feats_dir, feat_type)
    dump_pkl(features, feats_path)
    print 'FEAT file created! Path: {}'.format(feats_path)

    if params.do_skip_thoughts:
        print("Generating skip-thoughts...")
        import create_skip_vectors
        class ArgsFaker():
            captions_file = cap_path
            output_file = os.path.join(pkl_dir, 'skip_vectors.pkl')

        fake_args = ArgsFaker()
        create_skip_vectors.main(fake_args)


def _validate(args):
    if args.type not in args.feats_dir:
        print("FATAL : Requested feature type {}, but directories are something else:\tfeats_dir={}".format(args.type, args.feats_dir))
        sys.exit(0)


if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-s', '--seed', type=int, help="Random seed.", default=SEED, required=False)
    arg_parser.add_argument('-f','--feats_dir',dest ='feats_dir',type=str, required=True)
    arg_parser.add_argument('-j','--json_dir',dest ='json_dir',type=str,required=True)
    arg_parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str,required=True)
    arg_parser.add_argument('-type','--type',dest ='type',type=str, choices=['resnet', 'googlenet', 'nasnetalarge', 'resnet152', 'pnasnet5large', 'polynet', 'senet154'])
    arg_parser.add_argument('-t','--test',dest = 'test',type=int,default=0,
                            help='perform small unit test. If value 0 not unit test if greater than 0 gets a dataset with that numbers of videos')
    arg_parser.add_argument('-sp', '--splits', dest='splits', type=str, default='0.61,0.05,0.34',
                            help='Create validation and test datasets. Usage: floats delimited by commas, '
                                 'of the form Tr,Val. ex: {-s 0.60,0.40}. Off by default.', required=False)
    arg_parser.add_argument('-proc', '--protocol', dest='protocol', type=str, default='')
    arg_parser.add_argument('-from_pkl', '--from_pkl', dest='from_pkl', type=int, default=0,
                            help='If >=1, load features from pickle file instead of raw feature files.'
                                 'Note that this is negated if loading pre-existing pickle files.')
    arg_parser.add_argument('-st', '--do_skip_thoughts', dest='do_skip_thoughts', action='store_true', default=False)
    arg_parser.add_argument('-y', '--yao', dest='yao', action='store_true', default=False, help='Use Yao2015 split.')

    args = arg_parser.parse_args()

    np.random.seed(args.seed)

    if not len(sys.argv) > 1:
        print arg_parser.print_help()
        sys.exit(0)

    _validate(args)

    y2t(args)
