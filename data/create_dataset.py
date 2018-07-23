"""
Dataset creation helper. Use this to generate command lines for lots of datasets.
"""
import logging
import os
import argparse

from util import *

from datetime import datetime

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

possible_features = ['resnet', 'googlenet', 'nasnetalarge', 'resnet152', 'pnasnet5large', 'densenet121', 'polynet', 'senet154']

SEED = 9


# When inputting dir paths in dict, make sure trailing `/` is removed.
dataset_to_meta = {
    'mvad':
        {"data_dir": "mvad/pkls",
         "base": "mvad/"},
    'vtt16':
        {"data_dir": "vtt/pkls2016",
         "base": "vtt"},
    'vtt17':
        {"data_dir": "vtt/pkls2017",
         "base": "vtt"},
    'youtube2text':
        {"data_dir": "youtube2text/pkls_yao",
         "base": "youtube2text"},
    'mpii':
        {"data_dir": "mpii/full",
         "base": "mpii"},
    'lsmdc16':
        {"data_dir": "lsmdc16/pkls16",
         "base": "lsmdc16/"},
    'tacos':
        {"data_dir": "TACoS/pkls",
         "base": "TACoS/"},
    'trecvid':
        {"data_dir": "trecvid/pkls",
         "base": "trecvid"},
    }


feats_dir_prefix = "features_"
test_feats_dir_prefix = "features_testing_"
annots_dir_name = "annotations"


def create_commands(args, datasets, features):
    """
    Create command lines to generate dataset files.

    :param args:
    :param datasets:
    :param features:
    :return: nil
    """

    lines = set()
    main_lines = set()
    counting = 0

    for ds in datasets:
        for ft in features:
            if ds == 'mvad' or ds == 'mpii' or ds == 'lsmdc16' or ds == 'tacos':
                # single-caption take feats from feats_dir
                data_dir = dataset_to_meta[ds]["data_dir"]
            else:
                # multi caption take feats from pkl dir
                data_dir = dataset_to_meta[ds]["data_dir"] + '_' + ft

            if args.test:
                data_dir += '_ut{}'.format(args.test)

            data_dir = os.path.join(args.base_path, data_dir)

            base_dir = dataset_to_meta[ds]["base"]
            feat_dir = os.path.join(base_dir, feats_dir_prefix + ft)
            test_feat_dir = os.path.join(base_dir, test_feats_dir_prefix + ft)
            annots_dir = os.path.join(base_dir, annots_dir_name)

            for p in (feat_dir, annots_dir):
                if not os.path.isdir(p):
                    logger.warning("Did not find directory at {}.".format(p))

            if 'vtt' in ds:
                if not os.path.isdir(test_feat_dir):
                    logger.warning("Did not find directory at {}.".format(test_feat_dir))

            main_cmd = create_line(args.seed, ds, annots_dir, ft, data_dir, feat_dir, test_feat_dir, args.test, args.skip_thoughts)

            if main_cmd in main_lines:
                continue

            main_lines.add(main_cmd)

            lines.add(main_cmd)

    create_command_files(args, lines)


def create_command_files(args, lines):
    out_txt_path = os.path.join(args.out, 'commands.txt')
    with open(out_txt_path, 'w') as f:
        for l in lines:
            f.write(l)
            f.write('\n')

    logger.info("Created list of dataset creation commands at {}".format(out_txt_path))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    creation_args = ap.add_argument_group("Creation Args")
    creation_args.add_argument('-bp', '--base_path', help="Base path to prepend onto the dataset_to_meta dict values defined above", default="")
    creation_args.add_argument('-ds', '--dataset_select', help="Select a dataset rather than all.", nargs='+', required=False,default=None, choices=dataset_to_meta.keys())
    creation_args.add_argument('-fs', '--feature_select', help="Select a feature type...", nargs='+', required=False,default=None, choices=possible_features)
    creation_args.add_argument('-s', '--seed', help="Random seed", required=False, default=SEED)
    creation_args.add_argument('-t', '--test', help="Create unit-test dataset. 0=Off, otherwise size of unittest dataset, in samples.", default=0)
    creation_args.add_argument('-st', '--skip_thoughts', help="Perform skip-thoughts as SDM.", action='store_true', default=False)

    file_args = ap.add_argument_group("FileArgs")
    file_args.add_argument("-o", "--out", help="Output file for generated commands from this script.", required=True)

    args = ap.parse_args()

    mkdirs_safe(args.out)

    if args.dataset_select:
        if type(args.dataset_select) == list:
            datasets = args.dataset_select
        else:
            datasets = [args.dataset_select]
    else:
        datasets = dataset_to_meta.keys()

    if args.feature_select:
        if type(args.feature_select) == list:
            features = args.feature_select
        else:
            features = [args.feature_select]
    else:
        features = possible_features

    create_commands(args, datasets, features)
