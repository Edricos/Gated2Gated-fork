#!/bin/sh
export CUDA_VISIBLE_DEVICES=0
python src/train.py \
--data_dir                  ./data \
--log_dir                   ./logs \
--coeff_fpath               chebychev \
--depth_flat_world_fpath    depth_flat_world \
--model_name                multinetwork \
--model_type                multinetwork \
--exp_name                  multinetwork \
--models_to_load            depth ambient encoder albedo pose pose_encoder     \
--load_weights_folder       ./models/initialization \
--exp_num                   0 \
--height                    512 \
--width                     1024 \
--num_bits                  10 \
--scales                    0 \
--frame_ids                 0 -1 1 \
--pose_model_type           separate_resnet \
--num_layers                18 \
--weights_init              pretrained \
--pose_model_input          pairs \
--min_depth                 0.1 \
--max_depth                 100.0 \
--dataset                   gated \
--split                     gated2gated \
--batch_size                4 \
--num_workers               4 \
--learning_rate             2e-4 \
--num_epochs                20 \
--scheduler_step_size       15 \
--disparity_smoothness      0.001 \
--log_frequency             200 \
--save_frequency            1 \
--cycle_weight              0.05 \
--depth_normalizer          70.0 \
--passive_weight            0.01 \
--cycle_loss \
--temporal_loss \
--sim_gated \
--v1_multiscale \
--infty_hole_mask   \
--snr_mask \
--intensity_mask \
--passive_supervision \
