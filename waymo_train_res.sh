#!/bin/bash
python train.py \
--gpus_to_use 0 \
--data_path ../waymo_gendata/training \
--png \
--weighted_ssim \
--boxify \
--model_name ./waymo_resmof_W1/ \
--seg_mask none \
--log_frequency 2000 \
--save_frequency 2 \
--batch_size 12 \
--log_depth \
--log_lr \
--log_multiframe \
--log_trans whole \
--reconstr_loss_weight 0.85 \
--ssim_loss_weight 1.5 \
--smooth_loss_weight 1e-3 \
--motion_smooth_loss_weight 1 \
--motion_sparsity_loss_weight 1 \
--depth_consistency_loss_weight 1e-2 \
--rot_loss_weight 1e-3 \
--trans_loss_weight 2e-2 \
--use_norm_in_downsample \
--epoch_only_ego 0 \
--load_weights_folder models/waymo_ego_2/models/weights_3 \
--models_to_load encoder depth pose motion scaler \
--height 320 \
--width 480 

# --learning_rate 1e-4 
# --num_epochs 40 \
# --load_weights_folder models/ego_motion_weights_9 \