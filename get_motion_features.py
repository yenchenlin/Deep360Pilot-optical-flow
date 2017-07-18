import numpy as np
import glob
import cv2
import argparse
from joblib import Parallel, delayed

W = 1920
H = 960

n_filter = 5
motion_thresh = 0.2
inclusion_thresh = 0.1

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-d', '--domain', dest='domain')
    parser.add_argument('-b', '--boxes', dest='n_boxes', type=int)
    args = parser.parse_args()

    return args


def preprocess_box(xmin, ymin, xmax, ymax):
    if xmin < 0: xmin = 0
    elif xmin > W: xmin = int(xmin - W)
    if ymin < 0: ymin = 0
    elif ymin > H: ymin = int(ymin - H)
    if xmax > W: xmax = int(W)
    elif xmax <= xmin: xmax = xmin+1
    if ymax > H: ymax = int(H)
    elif ymax <= ymin: ymax = ymin+1
    return (xmin, ymin, xmax, ymax)


def motion_saliency(flow_mag, n):
    prior = flow_mag / np.max(flow_mag)
    filt = np.ones((n, n))/n/n

    likeli = cv2.filter2D(flow_mag.astype(np.float32), -1, filt)
    likeli = (likeli - likeli.min()) / (likeli.max() - likeli.min())

    return likeli * prior

def get_motion_features(name):
    print name

    FLOW_DIR = '/home/yenchen/data/of_' + args.domain + '/' + name + '/'
    BOXES_DIR = '/home/yenchen/data/feature_' + args.domain + \
        '_' + str(args.n_boxes) + 'boxes/' + name + '/'

    n_frames = len(glob.glob(FLOW_DIR + '*.png'))
    # init boxes
    clip_boxes_index = 1
    clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))

    #pruned_boxes = np.zeros(clip_boxes.shape)
    avg_motion = np.zeros((50, args.n_boxes))
    motion_inclusion = np.zeros((50, args.n_boxes))
    avg_flow = np.zeros((50, args.n_boxes, 2))

    for i in xrange(1, n_frames+1):
        print "{}, Flow {}, ".format(name, i)
        # boxes
        new_clip_boxes_index = (i-1) / 50 + 1
        if clip_boxes_index != new_clip_boxes_index:
            # 1.1 save pruned_boxes and init a new one
            #np.save(BOXES_DIR + 'pruned_boxes{:04d}.npy'.format(clip_boxes_index), pruned_boxes)
            #pruned_boxes = np.zeros(clip_boxes.shape)

            # 1.2 save avg motion and init a new one
            np.save(BOXES_DIR + 'avg_motion{:04d}.npy'.format(clip_boxes_index), avg_motion)
            avg_motion = np.zeros((50, args.n_boxes))

            # 1.3 save motion inclusion and init a new one
            np.save(BOXES_DIR + 'motion_inclusion{:04d}.npy'.format(clip_boxes_index), motion_inclusion)
            motion_inclusion = np.zeros((50, args.n_boxes))

            # 1.4 save avg flow and init a new one
            np.save(BOXES_DIR + 'avg_flow{:04d}.npy'.format(clip_boxes_index), avg_flow)
            avg_flow = np.zeros((50, args.n_boxes, 2))

            # 2.1 update clip_boxes
            clip_boxes_index = new_clip_boxes_index
            clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))

        # flow
        flow_img = np.array(cv2.imread(FLOW_DIR + '{:06d}.png'.format(i)), dtype=np.float32)
        flow_mag = flow_img[:, :, 1:] - 128
        flow_mag = flow_mag / 128

        # compute saliency
        motion_map = motion_saliency(flow_mag, n_filter)
        mot_thresh = min([motion_thresh, 0.7*np.max(motion_map)])
        motion_map = motion_map >= mot_thresh

        has_box = False
        frame_boxes = clip_boxes[(i-1) % 50].astype(int)
        for box_id, (xmin, ymin, xmax, ymax) in enumerate(frame_boxes):
            xmin, ymin, xmax, ymax = preprocess_box(xmin, ymin, xmax, ymax)
            box_motion_map = motion_map[ymin:ymax, xmin:xmax, :]
            inclusion = float(np.sum(box_motion_map)) / (np.sum(motion_map) + 1e-6)

            # average motion per box
            area = (xmax - xmin) * (ymax - ymin)
            box_motion_mag = np.sqrt(np.sum(np.square(box_motion_map), axis=2))
            box_avg_motion_mag = float(np.sum(box_motion_mag)) / area
            avg_motion[(i-1) % 50][box_id] = box_avg_motion_mag

            # motion inclusion per box
            motion_inclusion[(i-1) % 50][box_id] = inclusion

            # avg flow mag per box
            box_flow_mag = flow_mag[ymin:ymax, xmin:xmax]
            avg_flow[(i-1) % 50][box_id][0] = np.mean(box_flow_mag[:, :, 0])
            avg_flow[(i-1) % 50][box_id][1] = np.mean(box_flow_mag[:, :, 1])

            if inclusion >= inclusion_thresh:
                has_box = True
                #print "Box id {}, {} {} {} {}, area: {}, motion: {}".format(
                #    box_id, xmin, ymin, xmax, ymax, area, inclusion)
                #pruned_boxes[(i-1) % 50][box_id] = np.array([xmin, ymin, xmax, ymax])


    # save latest pruned_boxes
    #np.save(BOXES_DIR + 'pruned_boxes{:04d}.npy'.format(clip_boxes_index), pruned_boxes)

    # save latest motion magnitude
    np.save(BOXES_DIR + 'avg_motion{:04d}.npy'.format(clip_boxes_index), avg_motion)

    # save latest motion inclusion
    np.save(BOXES_DIR + 'motion_inclusion{:04d}.npy'.format(clip_boxes_index), motion_inclusion)

    # save latest flow mag
    np.save(BOXES_DIR + 'avg_flow{:04d}.npy'.format(clip_boxes_index), avg_flow)

if __name__ == '__main__':
    args = parse_args()
    NAMEs = sorted(np.load('/home/yenchen/Workspace/' + str(args.n_boxes) + \
                           '_boxes_data/metadata/metadata_' + args.domain + '.npy').item().keys())
    print NAMEs
    Parallel(n_jobs=5)(delayed(get_motion_features)(name) for name in NAMEs)
