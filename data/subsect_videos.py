import os
import sys
import json
import re
import argparse
from math import floor
from multiprocessing import Pool


def do_command(command):
    os.system(command)


def general_case(args):
    if args.annots_path.endswith('.json'):
        # Load user specified json file.
        json_file = open(args.annots_path)
    else:
        json_file = open(os.path.join(args.annots_path, 'videodatainfo_2017.json'))

    json_str = json_file.read()
    json_data = json.loads(json_str)

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    start = int(args.start)
    end = int(args.end)

    src_files = os.listdir(src_dir)


    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    command_list = []

    for video_file in src_files[start:end]:
        # Get index from video file name
        video_index = int(re.findall('\d+', video_file)[0])

        # Two scenarios:
        #       Subsecting training videos, which go video0 to video9999
        #       Subsecting test videos, which go video10000 to vieo12999
        # To account for either case, take mod 10000 to get the correct 0-based index to use in json lookup.
        video_index %= 10000

        start_time = float(json_data['videos'][video_index]['start time'])
        end_time = float(json_data['videos'][video_index]['end time'])
        duration = end_time - start_time

        src_path = os.path.join(src_dir, video_file)

        dst_path = os.path.join(dst_dir, video_file)

        if os.path.isfile(dst_path):
            print 'File at {} already exists!'.format(dst_path)
            continue

        ffmpeg_subsection_cmd = "ffmpeg -ss {} -i {} -t {} -vcodec copy -acodec copy {}".format(
            start_time, src_path, duration, dst_path)
        command_list.append(ffmpeg_subsection_cmd)

    threadPool = Pool()
    threadPool.map(do_command, command_list)
    threadPool.close()
    threadPool.join()


def tacos(args):
    def frame_to_timeestamp(frame_rate, frame_num):
        return float("%.3f" % (float(frame_num) / float(frame_rate)))

    if args.annots_path.endswith('.tsv'):
        # Load user specified json file.
        tsv_file = open(args.annots_path)
    else:
        tsv_file = open(os.path.join(args.annots_path, 'index.tsv'))

    data = [i for i in tsv_file]

    src_dir = args.src_dir
    dst_dir = args.dst_dir
    start = int(args.start)
    end = int(args.end)

    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    command_list = []

    for line in data:
        groups = line.replace('\n', '').split('\t')
        dest_vid = groups[0]
        sentence = groups[1]
        src_vid = groups[2]
        start_frame = float(groups[3])
        end_frame = float(groups[4])

        start_time = frame_to_timeestamp(29.40, start_frame)
        duration_time = frame_to_timeestamp(29.40, end_frame - start_frame)
        src_path = os.path.join(src_dir, src_vid + '.avi')
        dst_path = os.path.join(dst_dir, dest_vid + '.avi')

        if os.path.isfile(dst_path):
            print 'File at {} already exists!'.format(dst_path)
            continue

        ffmpeg_subsection_cmd = "ffmpeg -ss {} -i {} -t {} -vcodec copy -acodec copy {}".format(
            start_time, src_path, duration_time, dst_path)
        command_list.append(ffmpeg_subsection_cmd)

    threadPool = Pool()
    threadPool.map(do_command, command_list)
    threadPool.close()
    threadPool.join()


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('src_dir', help='directory where to get full videos')
    arg_parser.add_argument('dst_dir',help = 'directory where to store subsections')
    arg_parser.add_argument('annots_path', help='directory where annotations file is stored')
    arg_parser.add_argument('start',help = 'start video index')
    arg_parser.add_argument('end',help = 'end video index')
    arg_parser.add_argument('--dataset', help='dataset being worked on')

    args = arg_parser.parse_args()

    if args.dataset == 'tacos':
        tacos(args)
    else:
        general_case(args)
