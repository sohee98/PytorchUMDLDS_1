# Copyright reserved.

from __future__ import absolute_import, division, print_function

import argparse
from options import WildOptions
from trainers import WildTrainer
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.utils import args_validity_check

import pdb
method_zoo = { 'wild': (WildOptions, WildTrainer) }

# python train.py --data_path ./gen_data --png --learn_intrinsics --weighted_ssim --boxify --model_name ./model_saved

def main_worker(rank, world_size, method, gpus, dist_backend, unknown_args1):

    options = method_zoo[method][0]()
    opts, unknown_args2 = options.parse()
    args_validity_check(unknown_args1, unknown_args2)
    opts.rank = rank
    opts.world_size = world_size            # 1
    opts.gpus = gpus                        # [0]
    opts.dist_backend = dist_backend        # 'nccl'
    trainer = method_zoo[method][1](opts)   # ==WildTrainer(opts)

    # pdb.set_trace()
    trainer.train()     # trainers/base_trainer.py

if __name__ == "__main__":
    # Select METHODOLOGY
    method_initializer = argparse.ArgumentParser(description="Method Initializer")
    method_initializer.add_argument("--method",
                             type=str,
                             choices = ["wild"],
                             default="wild",
                             help="depth estimation methodology to use")
    # GPU options
    method_initializer.add_argument("--gpus_to_use",
                             nargs="+",
                             type=int,
                             default=[0],
                             help="gpu(s) used for training")
    method_initializer.add_argument("--dist_backend",
                             type=str,
                             default='nccl',
                             choices = ['nccl', 'gloo'],
                             help="torch distributed built-in backends")

    args, unknown_args1 = method_initializer.parse_known_args()
    method = args.method
    world_size = len(args.gpus_to_use)
    gpus = args.gpus_to_use
    dist_backend = args.dist_backend

    if world_size > 1: # multi-gpu training
        mp.spawn(main_worker,
                 args=(world_size, method, gpus, dist_backend, unknown_args1),
                 nprocs=world_size,
                 join=True)
    else:
        main_worker(0, world_size, method, gpus, dist_backend, unknown_args1)


'''
{'encoder': WildDepthEncoder(
  (encoder): WildResNet(
    (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    (bn1): RandomizedLayerNorm()
    (relu): ReLU(inplace=True)
    (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
    (layer1): Sequential(
      (0): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): RandomizedLayerNorm()
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): RandomizedLayerNorm()
      )
      (1): BasicBlock(
        (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    ...

    (avgpool): Sequential()
    (fc): Sequential()
  )
), 'depth': WildDepthDecoder(
  (depth_decoder): ModuleDict(
    (upconv_4_0): TransposeConv3x3(
      (conv): ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2))
      (nonlin): ReLU(inplace=True)
    )
    (upconv_4_1): WildConvBlock(
      (conv): Conv3x3(
        (pad): ReflectionPad2d((1, 1, 1, 1))
        (conv): Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1))
      )
      (nonlin): ReLU(inplace=True)
    )
    (upconv_3_0): TransposeConv3x3(
      (conv): ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2))
      (nonlin): ReLU(inplace=True)
    )
    ...

    (depthconv_0): Sequential(
      (0): Conv3x3(
        (pad): ReflectionPad2d((1, 1, 1, 1))
        (conv): Conv2d(16, 1, kernel_size=(3, 3), stride=(1, 1))
      )
      (1): Softplus(beta=1, threshold=20)
    )
  )
), 'pose': PosePredictionNet(
  (pads): ModuleDict(
    (pad1): ConstantPad2d(padding=(0, 1, 0, 1), value=0)
    (pad2): ConstantPad2d(padding=(0, 1, 0, 1), value=0)
    (pad3): ConstantPad2d(padding=(0, 1, 0, 1), value=0)
    (pad4): ConstantPad2d(padding=(0, 1, 0, 1), value=0)
    (pad5): ConstantPad2d(padding=(0, 1, 0, 1), value=0)
    (pad6): ConstantPad2d(padding=(1, 1, 0, 1), value=0)
    (pad7): ConstantPad2d(padding=(1, 1, 0, 1), value=0)
  )
  (conv1): Conv2d(6, 16, kernel_size=(3, 3), stride=(2, 2))
  (conv2): Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2))
  (conv3): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2))
  (conv4): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2))
  (conv5): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2))
  (conv6): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2))
  (conv7): Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2))
  (global_pooling): AdaptiveAvgPool2d(output_size=1)
  (relu): ReLU(inplace=True)
  (background_motion_conv): Conv2d(1024, 6, kernel_size=(1, 1), stride=(1, 1), bias=False)
), 'motion': MotionPredictionNet(
  (refinement0): Conv2d(6, 3, kernel_size=(1, 1), stride=(1, 1))
  (refinement1): RefinementLayer(
    (conv1): Conv2d(1027, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2_1): Conv2d(1027, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2_2): Conv2d(1024, 1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(2048, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (relu): ReLU(inplace=True)
  )
  ...

  (final_refinement): RefinementLayer(
    (conv1): Conv2d(9, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2_1): Conv2d(9, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv2_2): Conv2d(6, 6, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv3): Conv2d(12, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (relu): ReLU(inplace=True)
  )
), 'scaler': RotTransScaler(), 'intrinsics_head': IntrinsicsHead(
  (focal_length_conv): Conv2d(1024, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (offsets_conv): Conv2d(1024, 2, kernel_size=(1, 1), stride=(1, 1), bias=False)
  (softplus): Softplus(beta=1, threshold=20)
)}



'''