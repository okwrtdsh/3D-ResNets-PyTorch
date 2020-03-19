from __future__ import print_function, division
import os
import sys
import subprocess
from multiprocessing import Pool
from glob import glob


def ffmpeg_process(args):
    video_file_path, dst_directory_path = args
    if '.avi' not in video_file_path:
        return False
    if os.path.basename(video_file_path).startswith('_'):
        return False
    if os.path.basename(video_file_path).startswith('.'):
        return False
    try:
        if os.path.exists(dst_directory_path):
            if not os.path.exists(os.path.join(dst_directory_path, 'image_00001.jpg')):
                pass
                # subprocess.call('rm -r \"{}\"'.format(dst_directory_path), shell=True)
                # print('remove {}'.format(dst_directory_path))
                # os.mkdir(dst_directory_path)
            else:
                return False
        else:
            os.mkdir(dst_directory_path)
    except Exception as e:
        print(e, dst_directory_path)
        return False
    cmd = 'ffmpeg -i \"{}\" -s 171x128 \"{}/image_%05d.jpg\"'.format(
        video_file_path, dst_directory_path)
    print(cmd)
    subprocess.call(cmd, shell=True)
    print('\n')


def class_process(dir_path, dst_dir_path):
    video_files = sorted(glob(os.path.join(dir_path, '*/*/*.avi')))
    with Pool(32) as p:
        res = p.map(ffmpeg_process, [
            (path, path.replace('video', 'jpg').replace('.avi', '/'))
            for path in video_files])
    return res


if __name__ == "__main__":
    """
    ./video/01/BaseballPitch/v_BaseballPitch_g01_c01.avi
    ./jpg/01/BaseballPitch/v_BaseballPitch_g01_c01/image_00001.jpg
    """
    dir_path = sys.argv[1]
    dst_dir_path = sys.argv[2]

    class_process(dir_path, dst_dir_path)

