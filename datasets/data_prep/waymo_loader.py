# All Rights Reserved.

"""Classes to load KITTI and Cityscapes data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os
import sys
import shutil
import re

from absl import logging
import numpy as np
import imageio
from PIL import Image
import cv2

from lib.img_processing import image_resize

from .base_loader import (
    BaseLoader, 
    get_resource_path, 
    get_seq_start_end,
    natural_keys
)

import pdb

class WaymoRaw(BaseLoader):
    r"""
    Base dataloader. We expect that all loaders inherit this class.
    """

    def __init__(self,
                 dataset_dir,
                 split,
                 load_pose=False,
                 img_height=128,
                 img_width=416,
                 seq_length=3,
                 data_format='mono2',
                 mask='none',
                 batch_size=32,
                 threshold=0.5):
        super().__init__(dataset_dir, img_height, img_width,
                         seq_length, data_format, mask, batch_size, 
                         threshold)

        # frames_file = 'datasets/data_prep/waymo/test_image_0.txt'

        self.load_pose = load_pose
        self.cam_id = '0'
        # self.collect_static_frames(static_frames_file)
        # self.collect_train_frames(frames_file)
        self.collect_train_frames()

    # def collect_static_frames(self, static_frames_file):
    #     # pdb.set_trace()
    #     frames = []
    #     with open(static_frames_file, 'r') as f:
    #         names = f.readlines()
    #         for name in names:
    #             n = name.split('  ')
    #             if '\n' in n:
    #                 n = n[:-1]
    #             frames += n

    #     self.static_frames = []
    #     for fr in frames:
    #         # unused_date, drive, frame_id = fr.split(' ')
    #         # fid = '%.10d' % (np.int(frame_id[:-1]))
    #         for cam_id in self.cam_ids:
    #             # self.static_frames.append(drive + ' ' + cam_id + ' ' + fid)
    #             self.static_frames.append(cam_id + ' ' + fr)

    # def collect_train_frames(self, frames_file):
    def collect_train_frames(self):
        r"""
        Create a list of training frames.
        """
        all_frames = []
        frames_dir = os.path.join(self.dataset_dir, 'image_' + self.cam_id)
        frames = os.listdir(frames_dir)
        frames.sort()
        for fr in frames:
            frame_id = fr[:-4]
            sid = int(frame_id) // 1000 +1
            scene_id = '%.4d' % sid         # 0000, 0001
            all_frames.append(scene_id + ' ' + frame_id)

        # all_frames = []
        # with open(frames_file, 'r') as f:
        #     names = f.readlines()
        #     for name in names:
        #         n6 = name.split('  ')
        #         for n in n6:
        #             if '\n' in n:
        #                 n = n[:-1]
        #             frame_id = n[:-4]
        #             sid = int(frame_id) // 1000 +1
        #             scene_id = '%.4d' % sid         # 0000, 0001
        #             all_frames.append(scene_id + ' ' + frame_id)
        
        # pdb.set_trace()

        # all_frames = []
        # drive_dir = self.dataset_dir
        # if os.path.isdir(drive_dir):
        #     for cam in self.cam_ids:
        #         img_dir = os.path.join(drive_dir, 'image_' + cam)
        #         num_frames = len(glob.glob(img_dir + '/*[0-9].png'))
                # for i in range(num_frames):
                #     frame_id = '%.10d' % i      # 0000000001
                #     all_frames.append(cam + ' ' + frame_id)
        
        assert len(all_frames)>0, 'no data found in the dataset_dir'


        self.train_frames = all_frames
        self.num_train = len(self.train_frames)
        # pdb.set_trace()


    
    def is_valid_sample(self, target_index):
        r"""
        Check whether we can find a valid sequence around this frame.
        """
        # if target_index == 197 : 
        #     pdb.set_trace()
        num_frames = len(self.train_frames)
        target_scene_id, _ = self.train_frames[target_index].split(' ')
        start_index, end_index = get_seq_start_end(target_index, self.seq_length)
        # Check if the indices of the start and end are out of the range
        if start_index < 0 or end_index >= num_frames:
            return False

        # pdb.set_trace()

        start_scene_id, start_frame_id= self.train_frames[start_index].split(' ')
        # start_drive, start_cam_id, start_frame_id= self.train_frames[start_index].split(' ')
        end_scene_id, end_frame_id = self.train_frames[end_index].split(' ')
        # end_drive, end_cam_id, end_frame_id = self.train_frames[end_index].split(' ')

        # check frame continuity
        if self.data_format == 'mono2':
            if (int(end_frame_id) - int(start_frame_id)) != (self.seq_length-1):
                return False

        # Check if the scenes and cam_ids are the same 
        # if (target_drive == start_drive and target_drive == end_drive and
        #     cam_id == start_cam_id and cam_id == end_cam_id):
        #     return True
        # return False
        if (target_scene_id == start_scene_id and target_scene_id == end_scene_id):
            return True
        return False

        # return True

    def load_image_sequence(self, target_index):
        r"""
        Return a sequence with requested target frame.
        """
        if self.data_format == 'struct2depth':
            start_index, end_index = get_seq_start_end(
                target_index,
                self.seq_length,
            )
        elif self.data_format == 'mono2':
            start_index = end_index = target_index
        
        image_seq = []
        dynamic_map_seq = []
        target_outlier_ratio = 0.0
        for index in range(start_index, end_index + 1):
            scene_id, frame_id = self.train_frames[index].split(' ')
            # frame_id = self.train_frames[index]
            infos = {
                'cam_id': self.cam_id,
                'scene_id': scene_id,
                'frame_id': frame_id
            }
            img, intrinsics = self.load_image_raw(infos)

            if index == target_index:
                zoom_y = self.img_height / img.shape[0]
                zoom_x = self.img_width / img.shape[1]

            # Notice the default mode for RGB images is BICUBIC
            img = np.array(Image.fromarray(img).resize((self.img_width, self.img_height)))
            image_seq.append(img)

        return image_seq, zoom_x, zoom_y, intrinsics

    def load_pose_sequence(self, target_index):
        r"""
        Returns a sequence of pose vectors for frames around the target frame.
        """
        target_drive, _, target_frame_id = self.train_frames[target_index].split(' ')
        target_pose = self.load_pose_raw(target_drive, target_frame_id)
        start_index, end_index = get_seq_start_end(target_frame_id, self.seq_length)
        pose_seq = []
        for index in range(start_index, end_index + 1):
            if index == target_frame_id:
                continue
            drive, _, frame_id = self.train_frames[index].split(' ')
            pose = self.load_pose_raw(drive, frame_id)
            # From target to index.
            pose = np.dot(np.linalg.inv(pose), target_pose)
            pose_seq.append(pose)
        return pose_seq


    def load_example(self, target_index):
        r"""
        Return a sequence with requested target frame.        
        """
        # pdb.set_trace()
        example = {}
        # target_drive, target_cam_id, target_frame_id = (
            # self.train_frames[target_index].split(' ') )
        target_scene_id, target_frame_id = (self.train_frames[target_index].split(' '))
        target_cam_id = self.cam_id
        infos = {
            # 'drive': target_drive,
            'scene_id' : target_scene_id,
            'cam_id': self.cam_id
        }

        image_seq, zoom_x, zoom_y, intrinsics = (
                self.load_image_sequence(target_index)
                )
        # pdb.set_trace()
        
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        # example['folder_name'] = target_drive + '_' + target_cam_id + '/'
        example['folder_name'] = target_cam_id + '/' + target_scene_id + '/'
        example['file_name'] = target_frame_id
        if self.load_pose:
            pose_seq = self.load_pose_sequence(target_index)
            example['pose_seq'] = pose_seq
        return example

    def load_pose_raw(self, drive, frame_id):   
        date = drive[:10]
        pose_file = os.path.join(
            self.dataset_dir, date, drive, 'poses', frame_id + '.txt'
        )
        with open(pose_file, 'r') as f:
            pose = f.readline()
        pose = np.array(pose.split(' ')).astype(np.float32).reshape(3, 4)
        pose = np.vstack((pose, np.array([0, 0, 0, 1]).reshape((1, 4))))
        return pose

    def load_image_raw(self, infos):
        r"""
        Load an raw image given its id.
        """
        # drive = infos['drive']
        # scene_id = infos['scene_id']
        cam_id = infos['cam_id']
        frame_id = infos['frame_id']
        # date = drive[:10]
        img_file = os.path.join(
            self.dataset_dir, 
            'image_' + cam_id,
            frame_id + '.png'
            )
        # pdb.set_trace()
        img = imageio.imread(img_file)              # '../waymo_kitti/testing/image_0/1000000.png' (1280, 1920, 4)
        intrinsics = self.load_intrinsics(infos)

        return img, intrinsics

    def load_intrinsics(self, infos):
        r"""
        Load the intrinsic matrix given its id.
        """
        # drive = infos['drive']
        cam_id = infos['cam_id']
        frame_id = infos['frame_id']
        # date = drive[:10]
        # pdb.set_trace()
        calib_file = os.path.join(self.dataset_dir, 'calib', frame_id + '.txt')
        filedata = self.read_raw_calib_file(calib_file)
        tr = np.reshape(filedata['P2'], (3, 4))     
        intrinsics = tr[:3, :3]
        return intrinsics

    # From https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    def read_raw_calib_file(self, filepath):
        r"""
        Read in a calibration file and parse into a dictionary.
        """
        data = {}
        with open(filepath, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which we don't
                # care about.
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
