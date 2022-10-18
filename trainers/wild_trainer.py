import os
import re
import time
import json
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from tensorboardX import SummaryWriter
import math

# custom modules
import datasets
from networks import wild_nets
from .base_trainer import BaseTrainer
from lib.img_processing import normalize_image, normalize_trans
from lib.torch_layers import combine, make_randomized_layernorm, \
                             compute_projected_rotation, \
                             compute_projected_translation, \
                             compute_projected_pixcoords, \
                             matrix_from_angles, get_motion_sparsity_loss, \
                             get_motion_smooth_loss, get_motion_smooth_loss2, \
                             get_smooth_loss, weighted_average

import pdb

class WildTrainer(BaseTrainer):
    def __init__(self, options):
        super(WildTrainer, self).__init__(options)

        self._init_dilation(self.opt.foreground_dilation)

    def _init_depth_net(self):
        # pdb.set_trace()
        """Build the depth network"""
        # from resnet_encoder.py
        if self.opt.not_use_layernorm:
            self.models["encoder"] = wild_nets.WildDepthEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained",
                use_norm_in_downsample=self.opt.use_norm_in_downsample)
        else:
            self.models["encoder"] = wild_nets.WildDepthEncoder(
                self.opt.num_layers, self.opt.weights_init == "pretrained",
                norm_layer=make_randomized_layernorm(self.opt.layernorm_noise_rampup_steps),
                use_norm_in_downsample=self.opt.use_norm_in_downsample)

        self.models["encoder"].to(self.device)

        self.models["depth"] = wild_nets.WildDepthDecoder(
            self.models["encoder"].num_ch_enc, scales=self.opt.scales,
            use_mono2_arch=self.opt.use_mono2_depth_decoder)
        self.models["depth"].to(self.device)

    def _init_pose_net(self):
        """Build the pose network"""

        self.models["pose"] = wild_nets.PosePredictionNet(
            (self.opt.height, self.opt.width),
            num_input_images=2)

        self.models["pose"].to(self.device)

        self.models["motion"] = wild_nets.MotionPredictionNet(
                (self.opt.height, self.opt.width),
                self.models["pose"].num_input_images,
                self.models["pose"].bottleneck_dims)
        self.models["motion"].to(self.device)
        self.models["scaler"] = wild_nets.RotTransScaler()
        self.models["scaler"].to(self.device)

        if self.opt.learn_intrinsics:
            self.models['intrinsics_head'] = wild_nets.IntrinsicsHead()
            self.models['intrinsics_head'].to(self.device)
        # pdb.set_trace()

    def _init_optimizer(self):
        """Add non-freezed parameters in the model to the optimizer"""
        # list of dictionaries
        parameters_to_train = []

        for n in self.models:
            if n in self.opt.models_to_freeze:
                print(f"freezing {n} weights")
                for param in self.models[n].parameters():
                    param.requires_grad = False
            else:
                # TODO: check if pose still needs partial regularization
                if n == 'pose': # partial regularization 
                    normal_parameters = []
                    # list of parameters for special treatment
                    other_parameters = []

                    for name, params in self.models['pose'].named_parameters():
                        # layers whose names begin with conv
                        if re.match('conv', name.split('.')[0]):
                            normal_parameters += [params]
                        else:
                            other_parameters += [params]

                    parameters_to_train += [
                        {'params': normal_parameters,
                         'weight_decay': eval(f'self.opt.{n}_weight_decay')},
                        {'params': other_parameters,
                         'weight_decay': eval(f'self.opt.{n}_other_weight_decay')}
                        ]

                else:
                    try:
                        parameters_to_train += [
                            {'params': self.models[n].parameters(),
                             'weight_decay': eval(f'self.opt.{n}_weight_decay')}
                             ]
                    except:
                        print(f'use the default weight decay for {n} network')
                        parameters_to_train += [
                            {'params': self.models[n].parameters(),
                             'weight_decay': self.opt.weight_decay}
                             ]

        self.model_optimizer = optim.Adam(
            parameters_to_train,
            self.opt.learning_rate)

        scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer,
            self.opt.scheduler_step_size, self.opt.scheduler_gamma)

        self.model_lr_scheduler = scheduler 

    def _init_dilation(self, foreground_dilation):
        p = foreground_dilation * 2 + 1
        self.dilation = nn.MaxPool2d(p, stride=1, padding=foreground_dilation)
        self.dilation.to(self.device)

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        ''' inputs의 key (frame_idxs:[0,-1,1])
        ('K', 0)                # [12,4,4]
        ('inv_K', 0)            # [12,4,4]
        ('color', 0, 0)         # specified image [12,3,128,416]
        ('color', -1, 0)
        ('color', 1, 0)
        ('color_aug', 0, 0)     # augmentation
        ('color_aug', -1, 0)
        ('color_aug', 1, 0)
        '''

        outputs = {}

        self._update_depths(inputs, outputs)
        self._update_poses(inputs, outputs)
        # pdb.set_trace()
        losses = self.compute_losses(inputs, outputs)

        return outputs, losses
        ''' outputs의 key
        ('depth', 0, 0)             #('depth', frame_id, scale)
        ('depth', -1, 0)            # [12,1,128,416]
        ('depth', 1, 0)

        ('axisangle', -1, 0)        # rotation from target to source (source, target)
        ('translation', -1, 0)      # translation from target to source
        ('proj_rot', -1, 0)         # rot-related Transformation from source to target
        ('proj_trans', -1, 0)       # Trans-related Transformation from source to target
        ('res_trans', -1, 0)        # pixel-wise translation

        ('axisangle', 0, -1)        # [12,1,3]
        ('translation', 0, -1)      # [12, 3, 128, 416]
        ('proj_rot', 0, -1)         # [12,4,4]
        ('proj_trans', 0, -1)       # [12, 3, 128, 416]
        ('res_trans', 0, -1)        # [12, 3, 128, 416]

        ('axisangle', 0, 1)
        ('translation', 0, 1)
        ('proj_rot', 0, 1)
        ('proj_trans', 0, 1)
        ('res_trans', 0, 1)

        ('axisangle', 0, 1)
        ('translation', 0, 1)
        ('proj_rot', 0, 1)
        ('proj_trans', 0, 1)
        ('res_trans', 0, 1)     
        '''


    def _update_depths(self, inputs, outputs):
        """Save depths of all the frames to inputs"""
        for frame_id in self.opt.frame_ids:
            features = self.models['encoder'](
                    inputs['color_aug', frame_id, 0]
                    )
            depths = self.models['depth'](features)

            for scale in self.opt.scales:
                outputs[('depth', frame_id, scale)] = depths[('depth', scale)]

    def _update_poses(self, inputs, outputs):
        """Save relative poses of all the frames to inputs"""

        frame_ids = sorted(self.opt.frame_ids) # [-1, 0, 1]
        num_pairs = len(self.opt.frame_ids) - 1

        # from source to target 
        for source in range(num_pairs):
            target = source + 1

            sf = frame_ids[source] # source frame
            tf = frame_ids[target] # target frame

            self._predict_pose(inputs, outputs, sf, tf)
            self._predict_pose(inputs, outputs, tf, sf)

    def compute_losses(self, inputs, outputs):
        """Compute the final weighted loss to run backward propagation"""
        losses = {}
        frame_ids = sorted(self.opt.frame_ids) # -1, 0, 1
        num_pairs = len(self.opt.frame_ids) - 1
        for source in range(num_pairs):
            source_frame_id = frame_ids[source]
            target_frame_id = frame_ids[source+1]

            # call the corresponding functions twice without and with 
            # source and target frames being swapped
            self._compute_pairwise_consistency_loss(inputs, outputs,
                                                    source_frame_id,
                                                    target_frame_id, losses)
            self._compute_pairwise_consistency_loss(inputs, outputs,
                                                    target_frame_id,
                                                    source_frame_id, losses)

            self._compute_pairwise_motion_smoothness(inputs, outputs,
                                                     source_frame_id,
                                                     target_frame_id, losses)
            self._compute_pairwise_motion_smoothness(inputs, outputs,
                                                     target_frame_id,
                                                     source_frame_id, losses)

            self._compute_pairwise_motion_sparisty_loss(inputs, outputs,
                                                        source_frame_id,
                                                        target_frame_id, losses)        ## added
            self._compute_pairwise_motion_sparisty_loss(inputs, outputs,
                                                        target_frame_id,
                                                        source_frame_id, losses)

        self._compute_depth_smoothness(inputs, outputs, losses)

        total_loss = 0

        for i, name in enumerate(losses):
            if name in self.opt.losses_to_use:
                losses[name] *= eval(f'self.opt.{name}_weight')
                total_loss += losses[name]
            else:
                if i == 0:
                    print(f'{name} not used for training')

        losses['loss'] = total_loss

        return losses

    def _predict_pose(self, inputs, outputs, source_frame_id, target_frame_id):
        """Returns pose prediction for a single pair of frames

        estimate the pose from the source frame to the target frame? (comment: not from t->s?)
            by feeding [target image, source image]
        """

        features = [inputs['color_aug', target_frame_id, 0],
                    inputs['color_aug', source_frame_id, 0]]
        pose_inputs = torch.cat(features, 1)

        # axisangle and translation from target to source
        # pdb.set_trace()
        # axisangle, translation, bottleneck, features = (
        #         self.models["pose"](pose_inputs)
        #         )
        background_motion, bottleneck, features = (
            self.models["pose"](pose_inputs)
        )
        axisangle = background_motion[:,:3, 0, 0]       # rotation
        translation = background_motion[:,3:, :, :]     # 전체 움직임

        # replace K and invK
        if not self.opt.learn_intrinsics:
            K = inputs[('K', 0)]
            inv_K = inputs[('inv_K', 0)]
        else:
            K = self.models['intrinsics_head'](
                    bottleneck, self.opt.width, self.opt.height)
            inv_K = torch.inverse(K)
            # TODO: check if the updates are needed
            inputs[('K', 0)] = K
            inputs[('inv_K', 0)] = inv_K
            
        # pixel-wise translation
        res_trans = self.models['motion'](background_motion, features)

        if self.opt.only_ego_motion or self.epoch <=self.opt.epoch_only_ego:
            res_trans *= 1e-12

        # re-scale the pose and motion
        # pdb.set_trace()
        # print(axisangle[0])   
        axisangle = self.models['scaler'](axisangle, 'rot')         #[12, 3]
        # print(axisangle[0])
        # print(translation[0])
        translation = self.models['scaler'](translation, 'trans')   #[12,3,1,1]
        # print(translation[0])
        # print(res_trans[0])
        res_trans = self.models['scaler'](res_trans, 'trans')       #[12,3,128,416]
        # print(res_trans[0])

        # use source_frame segmentation mask, normal scale
        # mask should composed of 0 and 1 in float format
        if self.opt.seg_mask != 'none':
            obj_masked_t = inputs['objects_being_masked'].float()
            obj_masked_t = obj_masked_t[..., None, None, None]
            outputs['objects_being_masked_tensor'] = obj_masked_t
            use_res_trans = 1 - outputs['objects_being_masked_tensor']
            trans_field = self._compute_trans_field(
                    translation,
                    res_trans * use_res_trans.repeat(1, 3, 1, 1),
                    inputs['mask', source_frame_id, 0])
        else:
            trans_field = self._compute_trans_field(
                    translation, 
                    res_trans)
        '''
        # compute the normalized residual motion map, added by fei
        normalized_trans = self._normalize_motion_map(
                    res_trans, 
                    trans_field)
        to be compatiable with monodepth2
        '''
        axisangle = axisangle.unsqueeze(1)      #[12,3]>>[12,1,3]

        # rotation from target to source
        outputs[("axisangle", source_frame_id, target_frame_id)] = axisangle
        # translation from target to source
        outputs[("translation", source_frame_id, target_frame_id)] = trans_field

        # rot-related Transformation from source to target
        outputs[("proj_rot", source_frame_id, target_frame_id)] = (
                compute_projected_rotation(axisangle, K, inv_K)
                )
        # trans-related Transformation from source to target
        outputs[("proj_trans", source_frame_id, target_frame_id)] = (
                compute_projected_translation(trans_field, K)
                )

        if self.opt.log_trans != 'none':
            outputs[("res_trans", source_frame_id, target_frame_id)] = res_trans
            # log the normalized motion map
            # outputs[("norm_trans", source_frame_id, target_frame_id)] = normalized_trans


    def _compute_trans_field(self, global_trans, residual_trans, seg_mask=None):
        # pdb.set_trace()
        if seg_mask == None:
            combined_trans = global_trans + residual_trans
        else:
            combined_trans = global_trans + residual_trans * self.dilation(seg_mask)
        return combined_trans
    
    # compute the normalized residual motion map, added by fei
    # def _normalize_motion_map(self, res_trans, trans_field):
    #     norm = torch.square(trans_field, axis=(1,2,3), keep_dim=True).mean() * 3.0
    #     return res_trans / tf.sqrt(norm + 1e-12)

    def _compute_pairwise_consistency_loss(self, inputs, outputs,
                                           source_frame_id, target_frame_id,
                                           losses):
        """Compute rgb and motion field consistency losses of a pair of frames

        1. estimate projected coordinates onto the target frame from the 
           grid points of the source frame.
        2. reconstruct the contents of the source frame by sampling the actual
           contents of the target frame
        """

        source_scale = 0
        source_rgb = inputs[("color", source_frame_id, source_scale)]
        # trans and rot from target to source
        source_trans = outputs[("translation", source_frame_id, target_frame_id)]
        source_rot = outputs[("axisangle", source_frame_id, target_frame_id)]

        target_rgb = inputs[("color", target_frame_id, source_scale)]
        target_depth = outputs[("depth", target_frame_id, source_scale)]
        # trans and rot from source to target
        target_trans = outputs[("translation", target_frame_id, source_frame_id)]
        target_rot = outputs[("axisangle", target_frame_id, source_frame_id)]
        

        for scale in self.opt.scales:
            source_depth = F.interpolate(
                outputs[('depth', source_frame_id, scale)],
                [self.opt.height, self.opt.width],
                mode='bilinear',
                align_corners=True) #TODO: check True or False should be set
            proj_rot = outputs[('proj_rot', source_frame_id, target_frame_id)] 
            proj_trans = outputs[('proj_trans', source_frame_id, target_frame_id)] 

            # projected pix coords on the target frame
            # projected_depth of the target frame
            # (not estimated by the depth network)
            # mask represening the effectively projected coords
            pix_coords, projected_depth, effective_mask = (
                    compute_projected_pixcoords(
                        proj_rot, proj_trans,source_depth)
                    )
            
            # reconstructed source images by sampling the corresponding pixels
            # of the target frame
            target_rgb_resampled = F.grid_sample(
                target_rgb, pix_coords,
                padding_mode="zeros", align_corners=True)
            # reconstructed source depth by sampling the corresponding pixels
            # of the target frame
            target_depth_resampled = F.grid_sample(
                target_depth, pix_coords,
                padding_mode="zeros", align_corners=True)

            if scale == 0:
                # reconstructed source frame by the target frame
                outputs[('color', source_frame_id, target_frame_id)] = target_rgb_resampled
                outputs[('warp_mask', source_frame_id, target_frame_id)] = effective_mask
            '''
            ('color', -1, 0)
            ('warp_mask', -1, 0)
            ('color', 0, -1)
            ('warp_mask', 0, -1)
            ('color', 0, 1)
            ('warp_mask', 0, 1)
            ('color', 1, 0)
            ('warp_mask', 1, 0)

            '''

            # a mask to discriminate valid coordinates to compute
            # losses, not occluded and with effective projection
            # projected_depth >= target_depth_resampled represents where
            # occlusions may exist in the source frame
            # TODO add objects_being_masked_tensor into consideration
            if self.opt.seg_mask != 'none':
                # 1s for background; 0s for objects
                reversed_seg_mask = 1 - inputs['mask', source_frame_id, scale]
                # 1s for an input sequence being masked
                objs_being_masked_t = outputs['objects_being_masked_tensor']

                objs_not_masked = torch.clamp(
                        reversed_seg_mask + (1-objs_being_masked_t),
                        max=1.0)

                source_frame_closer_to_cam = torch.logical_and(
                        projected_depth < target_depth_resampled,
                        effective_mask,
                        )
                source_frame_closer_to_cam = torch.logical_and(
                        source_frame_closer_to_cam,
                        objs_not_masked==1).float()
            else:
                source_frame_closer_to_cam = torch.logical_and(
                        projected_depth < target_depth_resampled,
                        effective_mask
                        ).float()

            self._rgbd_consistency_loss(
                    projected_depth, effective_mask, source_rgb,
                    target_rgb_resampled, target_depth_resampled,
                    source_frame_closer_to_cam, losses)

            self._motion_field_consistency_loss(pix_coords,
                                                source_frame_closer_to_cam,
                                                source_rot, source_trans,
                                                target_rot, target_trans, losses)

    def _update_loss(self, losses, loss_name, add_value):
        """A helper function to update the value for a loss item"""
        try:
            losses[loss_name] += add_value
        except KeyError:
            losses[loss_name] = add_value

    def _rgbd_consistency_loss(self, source_projected_depth, effective_mask,
                               source_rgb, target_rgb_resampled,
                               target_depth_resampled,
                               source_frame_closer_to_cam, losses):
        """Compute depth and image consistency losses"""
        # aa=self._compute_l1_error(
        #             target_rgb_resampled, source_rgb,
        #             source_frame_closer_to_cam)

        # isNaN = math.isnan(aa.item())
        # if isNaN:
        #     pdb.set_trace()
        
        self._update_loss(
                losses, 'reconstr_loss', 
                self._compute_l1_error(
                    target_rgb_resampled, source_rgb,
                    source_frame_closer_to_cam)
                )
        self._update_loss(
                losses, 'depth_consistency_loss', 
                self._compute_l1_error(
                    target_depth_resampled, source_projected_depth,
                    source_frame_closer_to_cam)
                )

        if not self.opt.no_ssim: 
            if self.opt.weighted_ssim:
                squared_depth_diff = torch.square(
                        target_depth_resampled - source_projected_depth)

                depth_error_second_moment = weighted_average(
                        squared_depth_diff,
                        source_frame_closer_to_cam) + 1e-4

                # TODO: change effective_mask to source_frame_closer_to_cam
                depth_proximity_weight = torch.mul(
                        torch.div(depth_error_second_moment,
                                  squared_depth_diff + depth_error_second_moment),
                        effective_mask.float()
                        )
                # it is simply a running average
                # prevent gradients from propagating through this node
                # (bs, 1, h, w)
                depth_proximity_weight = depth_proximity_weight.detach()
                
                ssim_loss, avg_weight = self.ssim(target_rgb_resampled,
                                                  source_rgb,
                                                  depth_proximity_weight)
                ssim_loss = (ssim_loss * avg_weight).mean()
            else:
                ssim_loss = self.ssim(target_rgb_resampled, source_rgb).mean()

            self._update_loss(losses, 'ssim_loss', ssim_loss)

    def _motion_field_consistency_loss(self, projected_coords,
                                       source_frame_closer_to_cam,
                                       source_rot, source_trans,
                                       target_rot, target_trans, losses):
        """
        """
        # (bs, 3, 3) -> (bs, 1, 1, 3, 3)
        # rot-transformation from source to target
        source_rot_matrix = matrix_from_angles(source_rot)
        source_rot_matrix = source_rot_matrix.unsqueeze(1).unsqueeze(1)
        # rot-transformation from target to source
        target_rot_matrix = matrix_from_angles(target_rot)
        target_rot_matrix = target_rot_matrix.unsqueeze(1).unsqueeze(1)

        # from (bs, 1, h, w) to (bs, h, w)
        source_frame_closer_to_cam = source_frame_closer_to_cam.squeeze(1)

        # from (bs, 3, h, w) to (8, h, w, 3)
        source_trans = source_trans.permute(0,2,3,1)

        # stop gradients through projected_coords
        # TODO: check necessity of using stop gradients
        target_trans_resampled = F.grid_sample(
                target_trans, projected_coords.detach(),
                padding_mode="zeros", align_corners=True)

        target_trans = target_trans.permute(0,2,3,1)
        target_trans_resampled = target_trans_resampled.permute(0,2,3,1)

        # from (bs, 1, 1, 3, 3) to (bs, h, w, 3, 3)
        source_rot_matrix = source_rot_matrix.repeat(
                (1,) + source_trans.shape[1:3] +(1,1))
        target_rot_matrix = target_rot_matrix.repeat(
                (1,) + target_trans.shape[1:3] +(1,1))

        rot_unit, trans_zero = combine(
            target_rot_matrix, target_trans_resampled,
            source_rot_matrix, source_trans)
        eye = torch.eye(3, device=self.device).reshape(1,1,1,3,3)
        eye = eye.repeat(rot_unit.shape[:-2] + (1,1))

        rot_error = torch.square(rot_unit-eye).mean(dim=(3,4))
        rot1_scale = torch.square(source_rot_matrix-eye).mean(dim=(3,4))
        rot2_scale = torch.square(target_rot_matrix-eye).mean(dim=(3,4))
        rot_error /= (1e-24 + rot1_scale + rot2_scale)
        rot_loss = rot_error.mean()

        def norm(x):
            return torch.square(x).sum(dim=-1)

        # TODO: check how effective masks should be applied
        #       objects_being_masked, only for the denominator
        trans_loss = torch.div(
                torch.mul(source_frame_closer_to_cam, norm(trans_zero)),
                1e-24 + norm(source_trans) + norm(target_trans)
                ).mean()

        self._update_loss(losses, 'rot_loss', rot_loss)
        self._update_loss(losses, 'trans_loss', trans_loss)

    def _compute_pairwise_motion_smoothness(self, inputs, outputs,
                                            source_frame_id, target_frame_id,
                                            losses):
        # pdb.set_trace()
        motion_smooth_loss = get_motion_smooth_loss2(
                outputs[('res_trans', source_frame_id, target_frame_id)],       ## added
                outputs[('translation', source_frame_id, target_frame_id)]
                )
        self._update_loss(losses, 'motion_smooth_loss', motion_smooth_loss)

    def _compute_pairwise_motion_sparisty_loss(self, inputs, outputs,           ## added
                                           source_frame_id, target_frame_id,
                                           losses):
        motion_sparsity_loss = get_motion_sparsity_loss(
                outputs[('res_trans', source_frame_id, target_frame_id)],
                outputs[('translation', source_frame_id, target_frame_id)]
            )
        self._update_loss(losses, 'motion_sparsity_loss', motion_sparsity_loss)

    def _compute_depth_smoothness(self, inputs, outputs, losses):

        for scale in self.opt.scales:
            for frame_id in self.opt.frame_ids:
                disp = 1.0/outputs[('depth', frame_id, scale)]
                color = inputs[('color', frame_id, scale)] 
                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / mean_disp
                smooth_loss = get_smooth_loss(norm_disp, color)
                self._update_loss(losses, 'smooth_loss', smooth_loss)

    def post_warmup_init(self):
        """Reinitialize the trainer after the warmup epochs"""
        assert self.opt.save_frequency == 0
        self.opt.load_weights_folder = os.path.join(
                self.log_path, 'models', 'best')
        self.opt.models_to_load = ['encoder', 'depth', 'pose', 
                                   'intrinsics_head', 'scaler']
        self.opt.prob_to_mask_objects = 0.0
        # in order to load the pre-trained checkpoints
        self.opt.overwrite_old = False
        self.__init__(self.opt)

    def log(self, mode, inputs, outputs):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]

        if self.opt.log_multiscale:
            scales = self.opt.scales
        else:
            scales = [0]

        if self.opt.log_multiframe:
            frame_ids = self.opt.frame_ids
        else:
            frame_ids = [0]

        num_pairs = len(self.opt.frame_ids) - 1

        ## tensorboard image result
        # pdb.set_trace()

        for j in range(min(3, self.opt.batch_size)):
            # add reconstructed images
            for source in range(num_pairs):     # 2
                source_frame_id = sorted(self.opt.frame_ids)[source]        # [-1,0,1]
                target_frame_id = sorted(self.opt.frame_ids)[source+1]
                writer.add_image(
                    "color_pred_{}_{}/{}".format(source_frame_id, target_frame_id, j),
                    outputs[("color", source_frame_id, target_frame_id)][j].data, self.step)
                    # [3, 128, 416], self.step=3222
                writer.add_image(
                    "color_pred_{}_{}/{}".format(target_frame_id, source_frame_id, j),
                    outputs[("color", target_frame_id, source_frame_id)][j].data, self.step)
                writer.add_image(
                    "warp_mask_{}_{}/{}".format(source_frame_id, target_frame_id, j),
                    outputs[("warp_mask", source_frame_id, target_frame_id)][j].data, self.step)
                writer.add_image(
                    "warp_mask_{}_{}/{}".format(target_frame_id, source_frame_id, j),
                    outputs[("warp_mask", target_frame_id, source_frame_id)][j].data, self.step)
                # outputs[("color",-1,0)]           # [12, 3, 128, 416]
                # outputs[("warp_mask",-1,0)]       # [12, 1, 128, 416]
                # outputs[("color",0,-1)]           # [12, 3, 128, 416]
                # outputs[("warp_mask",0,-1)]       # [12, 1, 128, 416]
                # outputs[("color",0,1)]            # [12, 3, 128, 416]
                # outputs[("warp_mask",0,1)]        # [12, 1, 128, 416]
                # outputs[("color",1,0)]            # [12, 3, 128, 416]
                # outputs[("warp_mask",1,0)]        # [12, 1, 128, 416]

            for s in scales:    # [0]
                for frame_id in frame_ids:
                    writer.add_image(
                            "color_{}_{}/{}".format(frame_id, s, j),
                            inputs[("color", frame_id, s)][j].data, self.step)
                            # inputs[("color",0,0)]     # [12, 3, 128, 416]
                            # inputs[("color",-1,0)]    # [12, 3, 128, 416]
                            # inputs[("color",1,0)]     # [12, 3, 128, 416]

                    if self.opt.log_depth:
                        # pdb.set_trace()
                        writer.add_image(
                                "depth_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(
                                    outputs[("depth", frame_id, s)][j]),
                                self.step)

                    if self.opt.log_mobile_mask == 'normal':
                        writer.add_image(
                                "mask_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(
                                    inputs[("mask", frame_id, s)][j]),
                                self.step)
                    elif self.opt.log_mobile_mask == 'dilated':
                        writer.add_image(
                                "mask_{}_{}/{}".format(frame_id, s, j),
                                normalize_image(
                                    self.dilation(
                                        inputs[("mask", frame_id, s)][j]
                                        )
                                    ),
                                self.step)
                
                    writer.add_image(
                            "disp_{}_{}/{}".format(frame_id, s, j),
                            normalize_image(
                                1.0/outputs[("depth", frame_id, s)][j]),
                            self.step)
                    # outputs[("depth",0,0)]        # [12, 1, 128, 416]
                    # outputs[("depth",-1,0)]       # [12, 1, 128, 416]
                    # outputs[("depth",1,0)]        # [12, 1, 128, 416]

            # scale-independent info
            # pdb.set_trace()
            # for i in inputs: i
            if self.opt.log_trans != 'none':
                if self.opt.log_trans == 'masked':
                    mask = inputs[('mask', 0, 0)][j]
                else:                                   # log_trans == 'whole'
                    mask = 1 
                # pdb.set_trace()
                writer.add_image(
                        "translation/trans_{}_{}/{}".format(0, 1, j),
                        normalize_trans(outputs[("translation", 0, 1)][j]),
                        self.step)
                writer.add_image(
                        "translation/trans_{}_{}/{}".format(1, 0, j),
                        normalize_trans(outputs[("translation", 1, 0)][j]),
                        self.step)

                writer.add_image(
                        "translation/trans_res_{}_{}/{}".format(0, 1, j),
                        mask * normalize_trans(
                            outputs[("res_trans", 0, 1)][j]),
                        self.step)
                writer.add_image(
                        "translation/trans_res_{}_{}/{}".format(1, 0, j),
                        mask * normalize_trans(
                            outputs[("res_trans", 1, 0)][j]),
                        self.step)
                # outputs[("translation",0,1)]          # [12, 3, 128, 416]
                # outputs[("res_trans",0,1)]            # [12, 3, 128, 416]


    def log_test(self, mode):
        writer = self.writers[mode]

        self.set_eval()
        with torch.no_grad():
            # only use the 1st batch
            inputs = next(iter(self.test_loader))
            inputs = inputs.to(self.device)

            features = self.models['encoder'](inputs)
            depths = self.models['depth'](features)['depth', 0]
        self.set_train()
        
        for idx in range(min(4, self.opt.batch_size)):

            normalized_depth = normalize_image(depths[idx])
            normalized_disp = normalize_image(1.0/depths[idx])

            if self.opt.log_depth:
                writer.add_image(f"test_depth/{idx}",
                                 torch.cat([inputs[idx].data,
                                           normalized_depth.repeat(3,1,1)], 
                                           axis=1),
                                 self.step)
            writer.add_image(f"test_disp/{idx}",
                             torch.cat([inputs[idx].data,
                                       normalized_disp.repeat(3,1,1)], 
                                       axis=1),
                             self.step)

    def _weight_clipper(self):
        """ Clip model weights and parameters after optimization
        """
        if self.opt.world_size <= 1:
            self.models['scaler'].rot_scale.data = torch.clamp(
                    self.models['scaler'].rot_scale.data, min=0.001
                    )
            self.models['scaler'].trans_scale.data = torch.clamp(
                    self.models['scaler'].trans_scale.data, min=0.001
                    )
        else:
            self.models['scaler'].module.rot_scale.data = torch.clamp(
                    self.models['scaler'].module.rot_scale.data, min=0.001
                    )
            self.models['scaler'].module.trans_scale.data = torch.clamp(
                    self.models['scaler'].module.trans_scale.data, min=0.001
                    )

