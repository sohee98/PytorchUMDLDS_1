#!/bin/bash
python train.py \
--only_ego_motion \
--gpus_to_use 1 \
--data_path ../waymo_gendata/training  \
--png \
--weighted_ssim \
--boxify \
--model_name ./waymo_ego_2/ \
--seg_mask none \
--log_frequency 2000 \
--save_frequency 2 \
--batch_size 12 \
--log_depth \
--log_trans whole \
--reconstr_loss_weight 0.85 \
--ssim_loss_weight 1.5 \
--smooth_loss_weight 1e-3 \
--motion_smooth_loss_weight 1e-3 \
--depth_consistency_loss_weight 1e-2 \
--rot_loss_weight 1e-3 \
--trans_loss_weight 2e-2 \
--use_norm_in_downsample  \
--load_weights_folder models/kitti_ego2/models/weights_5 \
--models_to_load encoder depth pose motion scaler \
--height 320 \
--width 480 \
--num_workers 18 



# --log_multiframe \
# --log_lr \
# --data_path ../waymo_gendata/training \
