port=$(shuf -i 20000-30000 -n 1)
torchrun --nproc_per_node=2 --master_port ${port} demo.py \
 --output_dir "logs/sign_crowd_r50" \
 -c config/edpose.cfg.py \
 --sign_path ../data/ \
 --dataset_name MSASL \
 --split train \
 --options batch_size=8 epochs=80 lr_drop=75 num_body_points=14 backbone='resnet50' \
 --dataset_file="sign_video" \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_r50_crowdpose.pth" \
 --test \
 --kp_output_dir ../data/MSASL//keypoints_edpose_crowd_r50 \


