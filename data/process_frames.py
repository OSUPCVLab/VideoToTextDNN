import sys
import os
import argparse
import time
from multiprocessing import Pool


def main(args):
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    start = int(args.start)
    end = int(args.end)
    PREPEND = args.prepend

    src_files = os.listdir(src_dir)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    tuple_list = []

    for video_file in src_files[start:end]:
        src_path =  os.path.join(src_dir, video_file)
        dst_path = os.path.join(dst_dir, video_file)

        tuple_list.append((PREPEND, video_file, src_path, dst_path))

    pool = Pool() # Default to number cores
    pool.map(process_vid, tuple_list)
    pool.close()
    pool.join()


def process_vid(args):
    (PREPEND, video_file, src_path, dst_path) = args
    if not os.path.isdir(dst_path):
        os.mkdir(dst_path)
        # command = 'ffmpeg -i '+ src_path+' -s 256x256 '+ dst_path + '/%5d.jpg' #with resize
        command = PREPEND + 'ffmpeg -i '+ src_path+' -r 20 '+ dst_path + '/%6d.jpg > /dev/null 2>&1' #6 is to be in accordance with C3D features.
        print(command)

        os.system(command)
    else:
        print("Frames directory already found at {}".format(dst_path))


if __name__=='__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        'src_dir',
        help='directory where videos are'
    )
    arg_parser.add_argument(
        'dst_dir',
        help='directory where to store frames'
    )
    arg_parser.add_argument(
        'start',
        help='start index (inclusive)'
    )
    arg_parser.add_argument(
        'end',
        help='end index (noninclusive)'
    )
    arg_parser.add_argument(
        '--prepend',
        default='',
        help='optional prepend to start of ffmpeg command (in case you want to use a non-system wide version of ffmpeg)'
             'For example: --prepend ~/anaconda2/bin/ will use ffmpeg installed in anaconda2'
    )

    if not len(sys.argv) > 1:
        print(arg_parser.print_help())
        sys.exit(0)

    args = arg_parser.parse_args()

    start_time = time.time()
    main(args)
    print("Job took %s mins" % ((time.time() - start_time)/60))