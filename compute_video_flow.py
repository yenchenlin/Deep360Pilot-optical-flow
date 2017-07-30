import os
import glob
import argparse
from tqdm import trange
from subprocess import call
from flo2img import convert_wrapper
from joblib import delayed, Parallel

import sys
sys.path.append(os.path.abspath('.'))
import config as cfg

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    args = parser.parse_args()

    return args

args = parse_args()
DOMAIN = args.domain

DATA_PATH = cfg.FEATURE_PATH + '/'
EXE_PATH = os.path.join(os.getcwd(), 'Deep360Pilot-optical-flow')
NAMEs = sorted(map(lambda x: x.split('/')[-1], glob.glob(DATA_PATH + 'frame_' + DOMAIN + '/*')))
DEBUG = False

def compute_of(path, i):
	cmd = '{}/of {}{:06d} {}{:06d}'.format(EXE_PATH, path, i, path, i+1)
	call(cmd, shell=True)

if __name__ == '__main__':

    print DATA_PATH
    print DOMAIN
    print NAMEs
    cnt = 1
    for name in NAMEs:
	print name
        print "{}, {} / {}".format(name, cnt, len(NAMEs))
        path = DATA_PATH + 'frame_' + DOMAIN + '/' + name + '/'
        n_frames = len(glob.glob(path + '*.jpg'))
        print n_frames
        assert n_frames > 0, "No images in {}".format(path)

        print "Resizing images"
        cmd = 'mogrify -resize 25% -format ppm {}*.jpg'.format(path)
        call(cmd, shell=True)

        print "Extracting optical flow"
	Parallel(n_jobs=10)(delayed(compute_of)(path, i) for i in xrange(1, n_frames))
        #for i in trange(1, n_frames):
        #    cmd = './of {}{:06d} {}{:06d}'.format(path, i, path, i+1)
        #    call(cmd, shell=True)

        output_path = DATA_PATH + 'of_' + DOMAIN + '/' + name + '/'
        if not os.path.isdir(output_path):
            os.makedirs(output_path) # recursive mkdir

        convert_wrapper(path, output_path, DEBUG)

        cmd = 'mogrify -resize 400% {}*.png'.format(output_path)
        call(cmd, shell=True)

        cmd = 'rm -rf {}*.ppm'.format(path)
        call(cmd, shell=True)
