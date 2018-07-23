import argparse
import nltk

from util import *


SEED = 9


def get_annots_tacos(vid_feat_files, id_to_cap_dict, unittest, splits):
    vids_train = []
    vids_val = []
    vids_test = []
    all_vids = {}
    annotations = {}

    print 'Retrieving annotations...'
    if unittest:
        print 'UNIT TEST: On'
        n = unittest
    else:
        n = len(id_to_cap_dict)

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

    for enum, vid_id in enumerate(vid_feat_files):
        if unittest and enum > unittest:
            break

        cap = id_to_cap_dict[vid_id]

        if vid_id not in all_vids:
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
            cap_id = str(len(annotations[vid_id]))
            annotations[vid_id].append({'tokenized': tokenized, 'image_id': vid_id, 'cap_id': cap_id, 'caption': ocaption})
        else:
            annotations[vid_id]= []
            cap_id = str(0)
            annotations[vid_id].append({'tokenized': tokenized, 'image_id': vid_id, 'cap_id': cap_id, 'caption': ocaption})

        if count_train < num_train:
            vids_train.append(vid_id)
            count_train += 1
        elif count_valid < num_valid:
            vids_val.append(vid_id)
            count_valid += 1
        elif count_test < num_test:
            vids_test.append(vid_id)
            count_test += 1

    np.random.shuffle(vids_train)
    np.random.shuffle(vids_val)
    np.random.shuffle(vids_test)

    return annotations, vids_train, vids_val, vids_test, all_vids


def build_ground_truth_dict(gt_dir):
    csv_file = open(os.path.join(gt_dir, 'index.tsv'), 'r')

    id_to_cap_dict = {}
    for line in csv_file:
        groups = line.replace('\n', '').split('\t')
        dest_vid = groups[0]
        sentence = groups[1]

        # vidID -> sentence
        id_to_cap_dict[dest_vid] = sentence

    return id_to_cap_dict


def tacos(params):
    pkl_dir = params.pkl_dir
    feats_dir = params.feats_dir
    gt_dir = params.gt_dir
    unittest = params.test
    splits = params.splits

    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)

    train_path = os.path.join(pkl_dir, 'train.pkl')
    valid_path = os.path.join(pkl_dir, 'valid.pkl')
    test_path = os.path.join(pkl_dir, 'test.pkl')
    cap_path = os.path.join(pkl_dir, 'CAP.pkl')
    dict_path = os.path.join(pkl_dir, 'worddict.pkl')

    id_to_cap_dict = build_ground_truth_dict(gt_dir)
    vid_feat_files = os.listdir(feats_dir)

    annotations, vids_train, vids_val, vids_test, all_vids = get_annots_tacos(vid_feat_files, id_to_cap_dict, unittest, splits)

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


if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('-f', '--feats_dir', dest='feats_dir', type=str, default='')
    arg_parser.add_argument('-gt','--gt_dir',dest ='gt_dir',type=str, default='')
    arg_parser.add_argument('-p','--pkl_dir',dest ='pkl_dir',type=str, default='')
    arg_parser.add_argument('-t','--test',dest = 'test', type=int, default=0,
                            help='perform small unit test. If value 0 not unit test if greater than 0 gets a dataset with that numbers of videos')
    arg_parser.add_argument('-sp', '--splits', dest='splits', type=str, default='0.61,0.05,0.34',
                            help='Create validation and test datasets. Usage: floats delimited by commas, '
                                 'of the form Tr,Val. ex: {-s 0.60,0.20,0.20}. Default: 0.61,0.05,0.34')
    arg_parser.add_argument('-s', '--seed', type=int, help="Random seed.", default=SEED, required=False)
    arg_parser.add_argument('-st', '--do_skip_thoughts', dest='do_skip_thoughts', action='store_true', default=False)

    args = arg_parser.parse_args()

    np.random.seed(args.seed)

    if not len(sys.argv) > 1:
        print arg_parser.print_help()
        sys.exit(0)

    np.random.seed(args.seed)
    tacos(args)
