port=$(shuf -i 20000-30000 -n 1)
torchrun --nproc_per_node=2 --master_port ${port} demo.py \
 --output_dir "logs/sign_r50" \
 -c config/edpose.cfg.py \
 --sign_path ../data/ \
 --dataset_name MSASL \
 --split test \
 --options batch_size=8 epochs=60 lr_drop=55 num_body_points=17 backbone='resnet50' \
 --dataset_file="sign_video" \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_r50_coco.pth" \
 --test \
 --kp_output_dir ../data/MSASL/keypoints_edpose_coco_r50 \


