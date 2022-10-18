from params import par
import glob
import os
import numpy as np
import matplotlib.pyplot as plt

def visualize_pose():
    info = {'{:04d}'.format(i): [0, 39] for i in range(850)}
    for video in info.keys():
        fn = '{}{}.txt'.format(par.pose_dir, video)
        x = []
        y = []
        with open(fn) as f:
            print('processing file {}'.format(video), end='\r')
            lines = [line.split('\n')[0] for line in f.readlines()] 
            raw_poses = [ [float(value) for value in l.split(' ')] for l in lines ]

            for Rt in raw_poses:
                Rt = np.reshape(np.array(Rt), (3,4))
                t = Rt[:,-1]
                x.append(t[0])
                y.append(t[1])
        
        plot_name = '{}/visual/{}.jpg'.format(par.pose_dir, video)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'ro-')
        fig.suptitle('Pose number {}'.format(video))
        fig.savefig(plot_name)
        plt.close(fig)


visualize_pose()


