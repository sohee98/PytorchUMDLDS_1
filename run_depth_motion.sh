#!/bin/bash


DS=../waymo_gendata/training/0/0050
# DS=../waymo_gendata/training/0/0093


# MODEL=models/waymo_resmof_1/models/weights_15
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir ./output/waymo_resmof_1/weights_15/depth_motion \
# --model_path $MODEL

# MODEL=models/waymo_resmof_2.1/models/weights_15
# python infer_motion_mask_W.py \
# --input_path $DS \
# --output_dir output/waymo_resmof_2.1/weights_15/motion_mask \
# --model_path $MODEL

# MODEL=models/waymo_resmof_2.2/models/weights_15
# python infer_motion_mask_W.py \
# --input_path $DS \
# --output_dir output/waymo_resmof_2.2/weights_15/motion_mask \
# --model_path $MODEL


MODEL=models/waymo_resmof_3.1/models/weights_9
python infer_motion_mask_W.py \
--input_path $DS \
--output_dir output/waymo_resmof_3.1/weights_9/motion_mask \
--model_path $MODEL












# DS=gen_data_eigen/2011_09_26_drive_0051_sync_02
# python infer_depth_motion.py \
# --input_path $DS \
# --output_dir output/kitti_resmof2 \
# --model_path $MODEL

# MODEL=models/kitti_resmof2/models/weights_19

# DS=KITTI_test/2011_09_26_drive_0051_sync_02
# python infer_motion_mask_K.py \
# --input_path $DS \
# --output_dir output/kitti_resmof2/weight_5 \
# --model_path $MODEL
