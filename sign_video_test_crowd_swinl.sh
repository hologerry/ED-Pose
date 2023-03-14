export pretrain_model_path=../EDPose_pretrained/models/
port=$(shuf -i 20000-30000 -n 1)
torchrun --nproc_per_node=2 --master_port ${port} demo.py \
 --output_dir "logs/sign_crowd_swinl" \
 -c config/edpose.cfg.py \
 --sign_path ../data/ \
 --dataset_name MSASL \
 --split test \
 --options batch_size=8 epochs=80 lr_drop=75 num_body_points=14 backbone='swin_L_384_22k' \
 --dataset_file="sign_video" \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_swinl_crowdpose.pth" \
 --test \
 --kp_output_dir ../data/MSASL//keypoints_edpose_crowd_swinl \


