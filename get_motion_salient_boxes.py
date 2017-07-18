# This code is currently deprecated, use get_motion_inclusion.py instead

import numpy as np
import glob
import cv2
import argparse

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-n', '--name', dest='name')
    parser.add_argument('-d', '--domain', dest='domain')
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

if __name__ == '__main__':
    args = parse_args()

    FLOW_DIR = '/home/yclin/of_' + args.domain + '/' + args.name + '/'
    BOXES_DIR = '/home/yclin/feature_' + args.domain + '/' + args.name + '/'

    n_frames = len(glob.glob(FLOW_DIR + '*.png'))

    W = 1920
    H = 960

    n_filter = 5
    motion_thresh = 0.2
    motion_inclusion = 0.3
    area_thresh = 0.05

    # init boxes
    clip_boxes_index = 1
    clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))
    pruned_boxes = np.zeros(clip_boxes.shape)
    motion_inclusion = np.zeros((50, 16))

    for i in xrange(1, n_frames):
        print "Flow {}, ".format(i)
        # boxes
        new_clip_boxes_index = (i-1) / 50 + 1
        if clip_boxes_index != new_clip_boxes_index:
            # 1. save pruned_boxes and init a new one
            np.save(BOXES_DIR + 'pruned_boxes{:04d}.npy'.format(clip_boxes_index), pruned_boxes)
            pruned_boxes = np.zeros(clip_boxes.shape)

            # 2. update clip_boxes
            clip_boxes_index = new_clip_boxes_index
            clip_boxes = np.load(BOXES_DIR + 'roislist{:04d}.npy'.format(clip_boxes_index))

        # flow
        flow_img = np.array(cv2.imread(FLOW_DIR + '{:06d}.png'.format(i)), dtype=np.float32)
        flow_mag = flow_img[:, :, :2] - 128
        flow_mag = flow_mag / 128

        # compute saliency
        motion_map = motion_saliency(flow_mag, n_filter)
        mot_thresh = min([motion_thresh, 0.7*np.max(motion_map)])
        motion_map = motion_map >= mot_thresh

        frame_boxes = clip_boxes[(i-1) % 50].astype(int)
        for box_id, (xmin, ymin, xmax, ymax) in enumerate(frame_boxes):
            xmin, ymin, xmax, ymax = preprocess_box(xmin, ymin, xmax, ymax)
            box_motion_map = motion_map[ymin:ymax, xmin:xmax, :]
            inclusion = float(np.sum(box_motion_map)) / (np.sum(motion_map) + 1e-6)

            # box area
            area_rate = float((xmax - xmin) * (ymax - ymin)) / (W * H)
            inclusion = inclusion / area_rate * area_thresh

            #box_motion_mag = np.sqrt(np.sum(np.square(box_motion_map), axis=2))
            #box_avg_motion_mag = float(np.sum(box_motion_mag)) / area

            if inclusion >= motion_inclusion:
                print "Box id {}, {} {} {} {}, area: {}, motion: {}".format(
                    box_id, xmin, ymin, xmax, ymax, area_rate, inclusion)
                pruned_boxes[(i-1) % 50][box_id] = np.array([xmin, ymin, xmax, ymax])

    # save latest pruned_boxes
    np.save(BOXES_DIR + 'pruned_boxes{:04d}.npy'.format(clip_boxes_index), pruned_boxes)
