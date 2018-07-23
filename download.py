from bs4 import BeautifulSoup as Soup, SoupStrainer
import urllib
import os
import shutil
import json
import argparse
import sys
from multiprocessing import Pool


def download_mvad(command):
    os.system(command)


def video_mvad(args):
    dst_dir = args.dst_dir
    json_dir = args.json_path
    start = int(args.start)
    end = int(args.end)

    base_url = 'http://courvila_contact:59db938f6d@lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila'

    with open(os.path.join(json_dir, 'TrainList.txt'), 'r') as f:
        train_list = [i.replace('\n', '') for i in f]
    with open(os.path.join(json_dir, 'TestList.txt'), 'r') as f:
        test_list = [i.replace('\n', '') for i in f]
    with open(os.path.join(json_dir, 'ValidList.txt'), 'r') as f:
        valid_list = [i.replace('\n', '') for i in f]

    big_list = train_list + test_list + valid_list
    big_list = big_list[start:end]
    print "There are {} videos to get.".format(len(big_list))

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    present_vids = os.listdir(dst_dir)
    print "There are currently {} videos in dst_dir.".format(len(present_vids))

    count = 0

    if int(args.filter):
        print "FILTER: ON"
        filter_dir = os.path.join(dst_dir, '../trash/')
        if not os.path.isdir(filter_dir):
            os.makedirs(filter_dir)
        big_list_names = [i.split('/')[-1] for i in big_list]
        vids_to_move = []
        for i in present_vids:
            if i not in big_list_names:
                vids_to_move.append(i)

        for v in vids_to_move:
            print "Move {} -> {}".format(v, filter_dir)
            shutil.move(os.path.join(dst_dir, v), os.path.join(filter_dir, v))

        present_vids = os.listdir(dst_dir)
        print "There are now {} videos in dst_dir.".format(len(present_vids))

    command_list = []
    for i in big_list:
        video_name = i.split('/')[-1]
        if video_name not in present_vids:
            count += 1
            dst_path = os.path.join(dst_dir, video_name)
            #print video_name
            command_list.append('wget -O {} {}'.format(dst_path, base_url + i))

    threadPool = Pool()

    try:
        threadPool.map(download_mvad, command_list)
        threadPool.close()
        threadPool.join()
    except Exception:
        threadPool.close()
        threadPool.join()
        raise Exception


def video_mpii(video_dir,video_name,video_clip):


    # url='http://courvila_contact:59db938f6d@lisaweb.iro.umontreal.ca/transfert/lisa/users/courvila/data/lisatmp2/torabi/DVDtranscription/'+video_name+'/video/'+video_clip
    url='http://97H5:thoNohyee7@datasets.d2.mpi-inf.mpg.de/movieDescription/protected/avi/'+video_name+'/'+video_clip



    u2 = urllib.urlopen(url)
    video_dir_dst = os.path.join(video_dir,video_name)
    if not os.path.exists(video_dir_dst):
        os.mkdir(video_dir_dst)

    f = open(video_dir_dst+'/'+video_clip, 'wb')
    meta = u2.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (video_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u2.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,

    f.close()


def download_video((video_id, video_url)):
    dst_dir = args.dst_dir
    mp4_dst_path = "{}/{}.mp4".format(dst_dir, video_id)
    webm_dst_path = "{}/{}.webm".format(dst_dir, video_id)
    mkv_dst_path = "{}/{}.mkv".format(dst_dir, video_id)

    # Don't know the extension beforehand so check all of them
    if os.path.isfile(mp4_dst_path) or os.path.isfile(webm_dst_path) or os.path.isfile(mkv_dst_path):
        print 'File already downloaded!'
        return

    dst_path = "\'{}/{}.%(ext)s\'".format(dst_dir, video_id)
    cmd = "youtube-dl " + video_url + " -o {}".format(dst_path)
    os.system(cmd)


def video_vtt(args):

    def fill_info_list(videoID_to_info_tuple_list):
        if args.json_path.endswith('.json'):
            # Load user-specified json file
            json_file = open(args.json_path)
        else:
            json_file = open(os.path.join(args.json_path, 'videodatainfo_2017.json'))

        json_str = json_file.read()
        json_data = json.loads(json_str)

        start = int(args.start)
        end = int(args.end)  # Max vids to do

        for vid_meta in json_data['videos'][start:end]:
            video_id = vid_meta['video_id']
            video_url = vid_meta['url']

            videoID_to_info_tuple_list.append((video_id, video_url))

    dst_dir = args.dst_dir

    videoID_to_info_tuple_list = []

    fill_info_list(videoID_to_info_tuple_list)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    threadPool = Pool(1)  # Bottlenecked by network. Change to blank if otherwise
    threadPool.map(download_video, videoID_to_info_tuple_list)
    threadPool.close()
    threadPool.join()


def download_vine(command):
    print command
    os.system(command)


def video_trecvid(args):
    def fill_info_list(command_list):
        f = open(os.path.join(args.json_path, 'vines.url.testingSet'))

        start = int(args.start)
        end = int(args.end)  # Max vids to do
        dst_dir = args.dst_dir

        f = [l for l in f][start:end]

        for line in f:
            id, url = line.replace('\n', '').split('    ')
            dst_path = os.path.join(dst_dir, id + '.mp4')
            if not os.path.isfile(dst_path):
                command_list.append('wget -O {} {}'.format(dst_path, url))
            else:
                print "File already found! {}".format(dst_path)
    dst_dir = args.dst_dir

    command_list = []

    fill_info_list(command_list)

    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    threadPool = Pool()
    threadPool.map(download_vine, command_list)
    threadPool.close()
    threadPool.join()


if __name__== '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('dst_dir',help = 'directory where to store videos')
    arg_parser.add_argument('json_path', help='directory where json file is stored')
    arg_parser.add_argument('start',help = 'start video index')
    arg_parser.add_argument('end',help = 'end video index')
    arg_parser.add_argument('dataset', help = 'Which dataset to download. '
                                              'Options: vtt | trecvid | mvad')
    arg_parser.add_argument('--filter', help = 'Special mode which will filter out videos present in dst_dir but not in json file to dst_dir/../trash'
                                               'Options: 0 or 1', default=0)
    args = arg_parser.parse_args()

    if not len(sys.argv) > 1:
        print arg_parser.print_help()
        sys.exit(0)

    try:
        if args.dataset == 'vtt':
            video_vtt(args)
        elif args.dataset == 'trecvid':
            video_trecvid(args)
        elif args.dataset == 'mvad':
            video_mvad(args)
    except KeyboardInterrupt:
        print 'Interrupted'
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

