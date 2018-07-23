import os
import argparse
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def go(args=None, featsd=None, framesd=None):

    logger.info("\nParsing frame and feature directories.")
    if args is not None:
        feats_dir = args.feats_dir
        frames_dir = args.frames_dir
    else:
        feats_dir = featsd
        frames_dir = framesd

    feats = set(os.listdir(feats_dir))
    # '.'.join(i.split('.')[:-1]): Get video name up to the extension (last group)
    frames_to_ext = {'.'.join(i.split('.')[:-1]): i.split('.')[-1] for i in os.listdir(frames_dir)}
    frames = set(frames_to_ext.keys())

    logger.info('There are {} feature files and {} frame folders.'.format(len(feats), len(frames)))
    assert len(frames) >= len(feats)

    logger.info("Validate existing features...")
    bad_feats = set()
    invalid_paths = []
    sizes = {}

    for feat in feats:
        fpath = os.path.join(feats_dir, feat)
        stat = os.stat(fpath)
        sizes[fpath] = stat.st_size

        if stat.st_size <= 130:  # Empty npy file is usually 80 bytes. Flag file is 128
            bad_feats.add(feat)
            invalid_paths.append(fpath)

    if bad_feats:
        logger.warning("There are {} nil features.".format(len(bad_feats)))
        feats = feats - bad_feats
        logger.info("Invalid paths start:")
        for fpath in invalid_paths:
            print("-> " + fpath)
            if args.rm_nil:
                os.remove(fpath)
                print("--> Removed!")
    else:
        logger.info("Existing features are valid (filesize > 130B).")

    if sizes:
        logger.info("Smallest feature was {} Bytes\n------------------".format(min(sizes.values())))

    logger.info("In total, there are {} missing features.".format(len(frames - feats)))

    if args is None:
        needed_feats = frames - feats
        # Put back together extension since intersection is finished
        needed_feats = [i + '.' + frames_to_ext[i] for i in needed_feats]
        return needed_feats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('frames_dir', help='Frames directory')
    ap.add_argument('feats_dir', help='Features directory')
    ap.add_argument('-rm', '--rm_nil', help="Remove nil/invalid features.", default=False, action='store_true')

    args = ap.parse_args()

    go(args=args)
