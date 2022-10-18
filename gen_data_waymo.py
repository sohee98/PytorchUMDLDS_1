# Copyright All Rights Reserved.

"""Generates data for training/validation and save it to disk."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import multiprocessing
import os
import yaml
from absl import app
from absl import flags
from absl import logging
from torch.utils import data
import numpy as np
import imageio
import torch
import torchvision
from tqdm import tqdm

from options import DataGenOptions
from datasets import ProcessedImageFolder
from datasets.data_prep.video_loader import Video
from datasets.data_prep.kitti_loader import KittiRaw
from datasets.data_prep.waymo_loader import WaymoRaw

import pdb

FLAGS = DataGenOptions().parse()

NUM_CHUNKS = 100

def _generate_data():
    r"""
    Extract sequences from dataset_dir and store them in save_dir.
    """
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    if FLAGS.to_yaml:
        # Save the options to a YAML configuration in save_dir
        yaml_filename = os.path.join(FLAGS.save_dir, 'config.yaml')
        with open(yaml_filename, 'w') as f:
            yaml.dump(vars(FLAGS), f, default_flow_style=False)

    global dataloader  # pylint: disable=global-variable-undefined
    # import pdb; pdb.set_trace()
    if FLAGS.dataset_name == 'waymo':
        dataloader = WaymoRaw(
            FLAGS.dataset_dir,
            split='eigen',
            img_height=FLAGS.img_height,
            img_width=FLAGS.img_width,
            seq_length=FLAGS.seq_length,
            data_format=FLAGS.data_format,
            mask=FLAGS.mask, 
            batch_size=FLAGS.batch_size,
            threshold=FLAGS.threshold
      )
    else:
        raise ValueError('Unknown dataset')

    all_frames = range(dataloader.num_train)    # range(0, 29647)
    # Split into training/validation sets. Fixed seed for repeatability.
    np.random.seed(8964)

    num_cores = multiprocessing.cpu_count()     # 20
    # number of processes while using multiple processes
    # number of workers for using either a single or multiple processes
    num_threads = num_cores if FLAGS.num_threads is None else FLAGS.num_threads

    if FLAGS.single_process:
        frame_chunks = list(all_frames)
    else:
        frame_chunks = np.array_split(all_frames, NUM_CHUNKS)
        manager = multiprocessing.Manager()
        all_examples = manager.dict()
        pool = multiprocessing.Pool(num_threads)

    with open(os.path.join(FLAGS.save_dir, 'train_files.txt'), 'w') as train_f:
        with open(os.path.join(FLAGS.save_dir, 'val_files.txt'), 'w') as val_f:
            logging.info('Generating data...')

            for index, frame_chunk in enumerate(frame_chunks):
                if FLAGS.single_process:
                    all_examples = _gen_example(frame_chunk, {})
                    if all_examples is None:
                        continue
                else:
                    all_examples.clear()
                    # pdb.set_trace()
                    pool.map(
                        _gen_example_star,
                        zip(frame_chunk, itertools.repeat(all_examples))
                    )
                    logging.info(
                        'Chunk %d/%d: saving %s entries...', 
                        index + 1, NUM_CHUNKS, len(all_examples)
                    )
                # pdb.set_trace()
                for _, example in all_examples.items():
                    if example:
                        s = example['folder_name']
                        frame = example['file_name']
                        # print(s, frame)
                        if np.random.random() < 0.1:
                            val_f.write('%s %s\n' % (s, frame))
                        else:
                            train_f.write('%s %s\n' % (s, frame))
    train_f.close()
    val_f.close()

    if not FLAGS.single_process:
        pool.close()
        pool.join()

    if FLAGS.mask != 'none':
        # Collect filenames of all processed images
        img_dataset = ProcessedImageFolder(FLAGS.save_dir,
                                           FLAGS.save_img_ext)
        img_loader = torch.utils.data.DataLoader(
            img_dataset,
            batch_size=FLAGS.batch_size,
            num_workers=num_threads
        )

        # Generate masks by batch
        logging.info('Generating masks...')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for imgs, img_filepaths in tqdm(img_loader):
            mrcnn_results = dataloader.run_mrcnn_model(imgs.to(device))
            for i in range(len(imgs)):
                _gen_mask(mrcnn_results[i], img_filepaths[i], FLAGS.save_img_ext)

    if FLAGS.dataset_name=='video' and FLAGS.delete_temp:
        dataloader.delete_temp_images()
  
def _gen_example(i, all_examples=None):
    r"""
    Save one example to file.  Also adds it to all_examples dict.
    """
    add_to_file, example = dataloader.get_example_with_index(i)
    # print(add_to_file, example)
    if not example or dataloader.is_bad_sample(i):
        return
    # pdb.set_trace()
    ''' example => intrinsics' 'image_seq' 'folder_name' 'file_name'
    'folder_name': '0/1001/', 'file_name': '1000000'
    '''
    image_seq_stack = _stack_image_seq(example['image_seq'])
    example.pop('image_seq', None)  # Free up memory.
    intrinsics = example['intrinsics']
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    save_dir = os.path.join(FLAGS.save_dir, example['folder_name'])
    os.makedirs(save_dir, exist_ok=True)
    img_filepath = os.path.join(save_dir, f'{example["file_name"]}.{FLAGS.save_img_ext}')
    imageio.imsave(img_filepath, image_seq_stack.astype(np.uint8))
    # cam 저장
    cam_filepath = os.path.join(save_dir, '%s_cam.txt' % example['file_name'])
    example['cam'] = '%f,0.,%f,0.,%f,%f,0.,0.,1.' % (fx, cx, fy, cy)
    with open(cam_filepath, 'w') as cam_f:
        cam_f.write(example['cam'])

    # pdb.set_trace()
    if not add_to_file:
        return
    key = example['folder_name'] + example['file_name']
    all_examples[key] = example
    return all_examples

def _gen_example_star(params):
    return _gen_example(*params)

def _gen_mask(mrcnn_result, img_filepath, save_img_ext):
    f"""
    Generate a mask and save it to file.
    """
    mask_img = dataloader.generate_mask(mrcnn_result)
    mask_filepath = img_filepath[:-(len(save_img_ext)+1)] + f'-fseg.{save_img_ext}'
    imageio.imsave(mask_filepath, mask_img.astype(np.uint8))

def _gen_mask_star(params):
    return _gen_mask(*params)

def _stack_image_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res


if __name__ == '__main__':
    _generate_data()


'''
(Pdb) FLAGS
Namespace(augment_shift_h=0.0, 
augment_strategy='single', batch_size=4, 
crop=[0.0, 0.0, 0.0, 0.0], data_format='mono2', 
dataset_dir='../waymo_kitti/testing', 
dataset_name='waymo', 
del_static_frames=False, 
delete_temp=True, fps=10, 
img_height=128, img_width=416, 
intrinsics=None, mask='none', 
num_threads=None, 
save_dir='../waymo_gendata', s
ave_img_ext='png', 
seq_length=3, 
single_process=False, 
threshold=0.5, to_yaml=False, 
trim=[0.0, 0.0, 0.0, 0.0], video_end='00:00:00', video_start='00:00:00')

'''