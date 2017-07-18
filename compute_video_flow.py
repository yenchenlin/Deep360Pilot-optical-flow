import glob
import os
from subprocess import call
from tqdm import trange
from flo2img import convert_wrapper
from joblib import delayed, Parallel

DOMAIN = 'pilot'

DATA_PATH = '/home/yenchen/data/'
NAMEs = sorted(map(lambda x: x.split('/')[-1], glob.glob(DATA_PATH + 'frame_' + DOMAIN + '/*')))
DEBUG = False
def compute_of(path, i):
	cmd = './of {}{:06d} {}{:06d}'.format(path, i, path, i+1)
	call(cmd, shell=True)

if __name__ == '__main__':
    print NAMEs
    cnt = 1
    for name in NAMEs:
	print name
        print "{}, {} / {}".format(name, cnt, len(NAMEs))
        path = DATA_PATH + 'frame_' + DOMAIN + '/' + name + '/'
        n_frames = len(glob.glob(path + '*.jpg'))

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
            os.mkdir(output_path)

        convert_wrapper(path, output_path, DEBUG)

        cmd = 'mogrify -resize 400% {}*.png'.format(output_path)
        call(cmd, shell=True)

        cmd = 'rm -rf {}*.ppm'.format(path)
        call(cmd, shell=True)
