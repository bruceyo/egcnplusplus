# python tools/polyu_elderlyhome.py
import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap
#import polyu_read_skeleton
from tools.gen.polyu_read_skeleton import read_xyzang, read_ang, read_xyz_angw, read_xyz, read_xyzh, read_xyzhangw

max_body = 1
num_joint = 25
max_frame = 221
toolbar_width = 30

cs = {'1':[2,3,1,4,7],
      '2':[5,6,8,9,16],
      '3':[10,11,17,18,23],
      '4':[12,13,20,21,24],
      '5':[14,15,19,22,25]}

repetitions = np.loadtxt(fname = "./tools/gen/elderlyhome_repetitions.txt")

def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            action,
            benchmark='cv_cs',
            part='eval',
            fold='1',
            feature='both'):

    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):

        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 3])

        if action_class != action:
            continue

        subject_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 3])
        episode_id = int(
            filename[filename.find('L') + 1:filename.find('L') + 3])

        if_correct = 2

        if subject_id in [2,3,5,6,10,11,12,13,14,15]:
            if_correct = 1


        if benchmark == 'cv_cs':
            istraining = (subject_id not in cs[fold])
        elif benchmark == 'cv_rd':
            repetition = repetitions[subject_id-1][action_class-1]
            round = repetition // 5
            mod = repetition % 5
            training_list = []
            if int(fold) <= mod:
                training_list.append(int(5*round + int(fold)))
            for i in range(int(round)):
                training_list.append(i * 5 + int(fold))
            istraining = (episode_id not in training_list)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'eval':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(if_correct - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f, protocol=2)

    num_channel = 3
    if feature=='both':
        num_channel = 6

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), num_channel, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        if feature=='position':
            data = read_xyz(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        elif feature=='angle':
            data = read_ang(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)
        else:
            data = read_xyzang(
                os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)

        fp[i, :, 0:data.shape[1], :, :] = data
    end_toolbar()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='/skeleton')
    parser.add_argument(
        '--out_folder', default='/elderlyhome/')
    parser.add_argument('--joint_feature', default='angle', choices=['angle','position','both'], help='the feature of the skeleton data')

    benchmark = ['cv_rd', 'cv_cs']
    part = ['train', 'eval']
    fold = ['1', '2', '3', '4', '5']
    arg = parser.parse_args()

    if arg.joint_feature=='position':
        feature_folder = 'xyz'
    elif arg.joint_feature=='angle':
        feature_folder = 'ang'
    else:
        feature_folder = 'xyzang'

    for act in [1,2,3,4,5,6]:
        for b in benchmark:
            for f in fold:
                for p in part:
                    out_path = os.path.join(arg.out_folder, b, feature_folder, str(act), f)
                    if not os.path.exists(out_path):
                        os.makedirs(out_path)
                    print('Generate ', out_path)
                    gendata(
                        arg.data_path,
                        out_path,
                        act,
                        benchmark=b,
                        part=p,
                        fold=f,
                        feature=arg.joint_feature)

    # 1. line 118
    # 2. line 104
    # 3. line 114
