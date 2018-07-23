from sklearn.decomposition import PCA

import sys
import argparse
import numpy as np
import os
import shutil
from data import create_msr_vtt


def gather_feats(feats_dir, unittest):
    sampling = True

    feats_orig = os.listdir(feats_dir)
    if unittest:
        feats_orig = feats_orig[:unittest]

    assert len(feats_orig) >= 2

    # Get first feature so np.concatenate has something to use
    with open(os.path.join(feats_dir, feats_orig[0])) as f:
        feats = np.load(f)
        if sampling:
            feats = create_msr_vtt.get_sub_frames(feats)
    counter = 1

    for key in feats_orig[1:]:
        with open(os.path.join(feats_dir, key)) as f:
            feat = np.load(f)
            if sampling:
                feat = create_msr_vtt.get_sub_frames(feat)
            feats = np.concatenate((feats, feat), axis=0)
        sys.stdout.write('\r' + '{' + key + '} ' + str(counter) + '/' + str(len(feats_orig)) + '\n')
        sys.stdout.flush()
        counter+=1
    print "saving concatenated feats.."

    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--feats_dir', dest='feats_dir', type=str, default='')
    ap.add_argument('-ft', '--feats_testing_dir', dest='feats_testing_dir', type=str, default='')
    ap.add_argument('-pca', '--pca_dir', dest='pca_dir', type=str, default='')
    ap.add_argument('-pca_test', '--pca_test_dir', dest='pca_test_dir', type=str, default='')
    ap.add_argument('-type', '--type', dest='type', type=str, default='googlenet')
    ap.add_argument('-t', '--test', dest='test', type=int, default=0,
                    help='perform small unit test. If value 0 not unit test if greater than 0 gets a dataset with that numbers of videos')
    ap.add_argument('-train_pkl', '--training_pkl', dest='train_pkl', type=str, default='')
    ap.add_argument('-test_pkl', '--testing_pkl', dest='test_pkl', type=str, default='')

    if not len(sys.argv) > 1:
        print ap.print_help()
        sys.exit(0)

    args = ap.parse_args()

    feats_dir = args.feats_dir
    feats_test_dir = args.feats_testing_dir
    pca_dir = args.pca_dir
    pca_test_dir = args.pca_test_dir
    type = args.type
    unittest = args.test

    given_train_pkl = args.train_pkl
    given_test_pkl = args.test_pkl

    print "Extracting regular feature files..."
    extract_and_write_pca(feats_dir, feats_dir, pca_dir, type, unittest, given_train_pkl)

    #print "Extracting test feature files..."
    #extract_and_write_pca(feats_test_dir, feats_test_dir, pca_test_dir, type, unittest)


def extract_and_write_pca(transforming_feats_dir, fit_feats_dir, pca_dir, type, unittest, given_train_pkl):
    if given_train_pkl:
        pca = create_msr_vtt.load_pkl(given_train_pkl)
    else:
        # Refactor later to allow for mixing of fit feat files
        feats = gather_feats(fit_feats_dir, unittest)
        pca = PCA(n_components=1024).fit(feats)

    #dump_pkl(pca, os.path.join(pca_dir, 'pca_{}.pkl'.format(type)))

    if os.path.isdir(pca_dir):
        if raw_input("Found PCA folder, remove? [y/n]") == 'y':
            shutil.rmtree(pca_dir)
        else:
            print "Bye"
            sys.exit(0)

    os.mkdir(pca_dir)

    t_feat_files = os.listdir(transforming_feats_dir)
    if unittest:
        t_feat_files = t_feat_files[:unittest]

    for i, key in enumerate(t_feat_files, start=1):
        orig_feat_path = os.path.join(transforming_feats_dir, key)
        pca_feat_path = os.path.join(pca_dir, key)

        if type == 'c3d':
            feat = create_msr_vtt.load_c3d_feat(orig_feat_path)
            pca_feat = pca.transform(feat)

        elif type == 'resnet':
            with open(orig_feat_path) as f:
                feat = np.load(f)
                pca_feat = pca.transform(feat)
        else:
            print "Invalid feature type. Exiting."
            sys.exit(0)

        np.save(open(pca_feat_path, 'wb'), pca_feat)

        print str(i) + '/' + str(len(t_feat_files))
    print 'processed: ' + str(len(t_feat_files)) + " features."


if __name__ == '__main__':
    main()