import glob
import os
from subprocess import call
from tqdm import trange
from flo2img import convert_wrapper
from shutil import copyfile

NAMEs = sorted(map(lambda x: x[-13:], glob.glob('/home/yclin/frame_skate/*')))[:25]
DEBUG = False

if __name__ == '__main__':
    cnt = 1
    for name in NAMEs:
        print "{}, {} / {}".format(name, cnt, len(NAMEs))
        path = '/home/yclin/frame_skate/' + name + '/'
        n_frames = len(glob.glob(path + '*.jpg'))

        print "Resizing images"
        cmd = 'mogrify -resize 25% -format ppm {}*.jpg'.format(path)
        call(cmd, shell=True)

        print "Extracting optical flow"
        for i in trange(1, n_frames+1):
            if i == n_frames:
                copyfile('{}{:06d}'.format(path, i-1), '{}{:06d}'.format(path, i))
                continue
            cmd = './of {}{:06d} {}{:06d}'.format(path, i, path, i+1)
            call(cmd, shell=True)

        output_path = '/home/yclin/of_skate/' + name + '/'
        if not os.path.isdir(output_path):
            os.mkdir(output_path)

        convert_wrapper(path, output_path, DEBUG)

        cmd = 'mogrify -resize 400% {}*.png'.format(output_path)
        call(cmd, shell=True)

        cmd = 'rm -rf {}*.ppm'.format(path)
        call(cmd, shell=True)
