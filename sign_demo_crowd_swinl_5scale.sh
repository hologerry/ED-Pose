export pretrain_model_path=../EDPose_pretrained/models/
torchrun --nproc_per_node=2 demo.py \
 --output_dir "logs/sign_crowd_swinl_5scale" \
 -c config/edpose.cfg.py \
 --sign_path /D_data/SL/sign_data_processing/msasl_demo/images \
 --options batch_size=4 epochs=80 lr_drop=75 num_body_points=14 backbone='swin_L_384_22k' return_interm_indices=0,1,2,3 num_feature_levels=5 \
 --dataset_file="sign" \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_swinl_5scale_crowdpose.pth" \
 --test \
 --kp_output_dir /D_data/SL/sign_data_processing/msasl_demo/keypoints_edpose_crowd_swinl_5scale \


