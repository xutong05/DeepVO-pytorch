import os
import glob
import numpy as np
import time
from helper import R_to_angle
from helper import euler_from_matrix
from params import par
from torchvision import transforms
from PIL import Image
import torch
import math
from boston_singapore import boston_train, singapore

def clean_unused_images():
    seq_frame = {'00': ['000', '004540'],
                '01': ['000', '001100'],
                '02': ['000', '004660'],
                '03': ['000', '000800'],
                '04': ['000', '000270'],
                '05': ['000', '002760'],
                '06': ['000', '001100'],
                '07': ['000', '001100'],
                '08': ['001100', '005170'],
                '09': ['000', '001590'],
                '10': ['000', '001200']
                }
    for dir_id, img_ids in seq_frame.items():
        dir_path = '{}{}/'.format(par.image_dir, dir_id)
        if not os.path.exists(dir_path):
            continue

        print('Cleaning {} directory'.format(dir_id))
        start, end = img_ids
        start, end = int(start), int(end)
        for idx in range(0, start):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
            if os.path.isfile(img_path):
                os.remove(img_path)
        for idx in range(end+1, 10000):
            img_name = '{:010d}.png'.format(idx)
            img_path = '{}{}/{}'.format(par.image_dir, dir_id, img_name)
            if os.path.isfile(img_path):
                os.remove(img_path)


# transform poseGT [R|t] to [theta_x, theta_y, theta_z, x, y, z]
# save as .npy file
def create_pose_data():
    info = {'{:04d}'.format(i): [0, 39] for i in range(850)}

    start_t = time.time()
    for video in info.keys():
        fn = '{}{}.txt'.format(par.pose_dir, video)
        print('Transforming {}...'.format(fn))
        with open(fn) as f:
            lines = [line.split('\n')[0] for line in f.readlines()]

            #Different with Lei Lai
            raw_poses = [ [float(value) for value in l.split(' ')] for l in lines ]
            Rt0 = np.reshape(np.array(raw_poses[0]), (3,4))
            t0 = Rt0[:,-1]
            R0 = Rt0[:,:3]

            poses = [R_to_angle(Rt, t0, R0) for Rt in raw_poses] # list of pose (pose=list of 12 floats)
            poses = np.array(poses)
            base_fn = os.path.splitext(fn)[0]
            np.save(base_fn+'.npy', poses)
            print(np.reshape(poses[0, 6:15], (3, 3)))
            print('Video {}: shape={}'.format(video, poses.shape))
    print('elapsed time = {}'.format(time.time()-start_t))


def calculate_rgb_mean_std(image_path_list, minus_point_5=False):
    n_images = len(image_path_list)
    cnt_pixels = 0
    print('Numbers of frames in training dataset: {}'.format(n_images))
    mean_np = [0, 0, 0]
    mean_tensor = [0, 0, 0]
    to_tensor = transforms.ToTensor()

    image_sequence = []
    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        cnt_pixels += img_as_np.shape[1]*img_as_np.shape[2]
        for c in range(3):
            mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
            mean_np[c] += float(np.sum(img_as_np[c]))
    mean_tensor =  [v / cnt_pixels for v in mean_tensor]
    mean_np = [v / cnt_pixels for v in mean_np]
    print('mean_tensor = ', mean_tensor)
    # print('mean_np = ', mean_np)

    std_tensor = [0, 0, 0]
    std_np = [0, 0, 0]
    for idx, img_path in enumerate(image_path_list):
        print('{} / {}'.format(idx, n_images), end='\r')
        img_as_img = Image.open(img_path)
        img_as_tensor = to_tensor(img_as_img)
        if minus_point_5:
            img_as_tensor = img_as_tensor - 0.5
        img_as_np = np.array(img_as_img)
        img_as_np = np.rollaxis(img_as_np, 2, 0)
        for c in range(3):
            tmp = (img_as_tensor[c] - mean_tensor[c])**2
            std_tensor[c] += float(torch.sum(tmp))
            tmp = (img_as_np[c] - mean_np[c])**2
            std_np[c] += float(np.sum(tmp))
    std_tensor = [math.sqrt(v / cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / cnt_pixels) for v in std_np]
    print('std_tensor = ', std_tensor)
    # print('std_np = ', std_np)


if __name__ == '__main__':
    # clean_unused_images()
    create_pose_data()
    
    # Calculate RGB means of images in training videos
    train_video = ['{:04d}'.format(i) for i in boston_train]

    # print(train_video)
    image_path_list = []
    for folder in train_video:
        image_path_list += glob.glob('nuscenes/images_front/{}/*.jpg'.format(folder))

    calculate_rgb_mean_std(image_path_list, minus_point_5=True)