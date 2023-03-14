torchrun --nproc_per_node=2 demo.py \
 --output_dir "logs/sign_crowd_r50" \
 -c config/edpose.cfg.py \
 --sign_path /D_data/SL/sign_data_processing/msasl_demo/images \
 --options batch_size=4 epochs=80 lr_drop=75 num_body_points=14 backbone='resnet50' \
 --dataset_file="sign" \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_r50_crowdpose.pth" \
 --test \
 --kp_output_dir /D_data/SL/sign_data_processing/msasl_demo/keypoints_edpose_crowd_r50 \


