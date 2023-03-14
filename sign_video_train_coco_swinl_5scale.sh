export pretrain_model_path=../EDPose_pretrained/models/
port=$(shuf -i 20000-30000 -n 1)
torchrun --nproc_per_node=2 --master_port ${port} demo.py \
 --output_dir "logs/coco_swinl" \
 -c config/edpose.cfg.py \
 --options batch_size=8 epochs=60 lr_drop=55 num_body_points=17 backbone='swin_L_384_22k' return_interm_indices=0,1,2,3 num_feature_levels=5 \
 --dataset_file="sign_video" \
 --sign_path ../data/ \
 --dataset_name MSASL \
 --split train \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_swinl_5scale_coco.pth" \
 --test \
 --kp_output_dir ../data/MSASL/keypoints_edpose_coco_swinl_5scale \

