__author__ = 'onina'

import argparse
import json
import os
import pickle
import numpy as np

# import process_frames
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from multiprocessing import Pool
# import subprocess

from data.util import mkdirs_safe


def resizeImage(new_frame_path):
    print 'resizeImage: {}'.format(new_frame_path)
    command = 'magick {} -resize 1000x562\\! {}'.format(new_frame_path, new_frame_path)
    os.system(command)


def drawOverlay(image, text):

    print 'drawOverlay: ' + image + ' text: ' + text
    img = Image.open(image)
    draw = ImageDraw.Draw(img)
    text_length = len(text)*10

    w = 1000
    y = 440

    font_size = 16
    if text_length >= w:
        font_size = 12
        text_length = len(text) * 7.2

    x = int((w - text_length) / 2)
    draw.rectangle((x,y, x+text_length,y+16*1.5), fill=(0,0,0))

    font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    draw.text((x, y),' '+text+' ',(0,255,0),font=font)
    img.save(image)


def createVideo(inputdir, output_file, framerate):
    print 'createVideo: {} -> {}'.format(inputdir, output_file)
    command = 'magick -delay 7x100 -loop 0 ' + inputdir + '/*.jpg -layers OptimizePlus ' + output_file
    print 'command: {}'.format(command)
    os.system(command)
    print 'Finished job for {}.'.format(output_file)


def main(args):
    # print params
    with open(args.rfile) as file:
        data = json.load(file)

    needed_files = os.listdir(args.vidpath)

    id_to_file_dict = {}

    if args.dataset != 'other':
        real_to_id_dict = pickle.load(open(os.path.join(args.dict_path)))

    # Hack to make lsmdc look like the other json files.
    if args.dataset == 'lsmdc16':
        data = {'result': data}
        print("Cut to {} videos".format(args.test))
        print("Converted LSMDC16 to usable format")

    if args.test:
        data = {'result': np.random.choice(data['result'], args.test)}

    for i, desc in enumerate(data['result']):

        id = desc['video_id']
        found_file = False
        print("Attempt {}".format(id))

        for file in needed_files:
            if found_file:
                continue

            file_id = '.'.join(file.split('.')[:-1])
            if args.dataset != 'other':
                if file_id not in real_to_id_dict:
                    # print("{} not found in mapping dict.".format(file_id))
                    continue
                    
                if real_to_id_dict[file_id] == id:
                    id_to_file_dict[id] = file
                    found_file = True
            elif file_id == id:
                id_to_file_dict[id] = file
                found_file = True

        if not found_file:
            print "Didn't find the file for {}.".format(id)

    getting_frames_jobs = []
    video_create_jobs_after_frame_get = []
    video_create_jobs = []

    for i, desc in enumerate(data['result']):

        if desc['video_id'] not in id_to_file_dict.keys():
            continue

        if desc['caption'] == '':
            continue

        # print result['video_id']
        # print result['caption']

        processed_vid_path = os.path.join(args.vidpath, id_to_file_dict[desc['video_id']])

        out_path = os.path.join(args.rpath, 'frames')
        if not os.path.exists(out_path):
            mkdirs_safe(out_path)

        # out_path/out_frames_path.xxx/-> frames
        out_frames_path = os.path.join(out_path, id_to_file_dict[desc['video_id']])

        framerate = 30

        if not os.path.exists(out_frames_path):
            if os.path.isdir(processed_vid_path):
                getting_frames_jobs.append((desc, out_frames_path, out_path, processed_vid_path))

                rvid_path = os.path.join(args.rpath, 'vids')
                check_rvid_path(rvid_path)

                video_create_jobs_after_frame_get.append((desc, framerate, out_frames_path, rvid_path))
            else:
                print 'No processed video directory found at {}!'.format(processed_vid_path)
        else:
            print "final frames already created"
            rvid_path = os.path.join(args.rpath, 'vids')
            check_rvid_path(rvid_path)

            video_create_jobs.append((desc, framerate, out_frames_path, rvid_path))


    threadPoolWhenFramesNil = Pool()
    threadPoolFramesExisted = Pool()

    # Do frame getting jobs
    threadPoolWhenFramesNil.map(copy_frames_and_draw_overlay, getting_frames_jobs)
    # Also do video create jobs for frames already there
    threadPoolFramesExisted.map(prepare_path_and_create_video, video_create_jobs)

    threadPoolWhenFramesNil.close()
    threadPoolWhenFramesNil.join()

    threadPoolWhenFramesNil = Pool()

    # Now do video create jobs for previously nil frames
    threadPoolWhenFramesNil.map(prepare_path_and_create_video, video_create_jobs_after_frame_get)

    threadPoolWhenFramesNil.close()
    threadPoolWhenFramesNil.join()
    threadPoolFramesExisted.close()
    threadPoolFramesExisted.join()


def check_rvid_path(rvid_path):
    if not os.path.exists(rvid_path):
        mkdirs_safe(rvid_path)


def prepare_path_and_create_video((desc, framerate, out_frames_path, rvid_path)):
    new_vid_path = os.path.join(rvid_path, str(desc['video_id']) + '.gif')
    if not os.path.exists(new_vid_path):
        createVideo(out_frames_path, new_vid_path, framerate)


def copy_frames_and_draw_overlay((desc, out_frames_path, out_path, processed_vid_path)):
    print 'Copying files {} -> {}'.format(processed_vid_path, out_path)
    command = "cp -r {} {}".format(processed_vid_path, out_path)
    os.system(command)
    frames = os.listdir(out_frames_path)
    print 'Creating Image Overlay For ' + str(len(frames)) + ' frames'
    for frame in frames:
        if frame.endswith('.jpg') or frame.endswith('.png'):
            new_frame_path = os.path.join(out_frames_path, frame)
            print new_frame_path
            print desc['caption']
            resizeImage(new_frame_path)
            drawOverlay(new_frame_path, desc['caption'])


def _validate(args):
    if args.dataset == 'msvd' or args.dataset == 'lsmdc16':
        if not args.dict_path:
            raise ValueError("Was given dataset={} but no annotations path was given.".format(args.dataset))
        if not os.path.exists(args.dict_path):
            raise IOError("Was given dataset={} but dict_path={} does not exist.".format(args.dataset, args.dict_path))


if __name__=="__main__":
    #Run the script twice, the first time it will extract the frames the second time it will create the vids
    parser = argparse.ArgumentParser()
    parser.add_argument('-j','--rfile',dest='rfile', type=str, help='json file path with results',default='')
    parser.add_argument('-p','--vidpath',dest='vidpath',type=str,help='path where the processed videos reside', default='')
    parser.add_argument('-r','--rpath', dest='rpath',type=str, help='path where we will save vids',default='')
    parser.add_argument('-d', '--dataset', help="Dataset being processed. Some json files are written differently depending on the dataset.", default='other', choices=['msvd', 'lsmdc16', 'other'])
    parser.add_argument('-dp', '--dict_path', help="Path to msvd or lsmdc name mapping pkl file (containing mapping dict) [msdv & lsmdc16 only]", required=False)
    parser.add_argument('-t', '--test', type=int, help="Unit test/limit number movies to create to given arg. Default=0/OFF", default=0)
    parser.add_argument('-s', '--seed', help="Random seed.", default=9)

    args = parser.parse_args()
    np.random.seed(int(args.seed))

    _validate(args)
    main(args)
