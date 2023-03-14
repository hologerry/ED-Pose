export pretrain_model_path=../EDPose_pretrained/models/
  torchrun --nproc_per_node=2 demo.py \
 --output_dir "logs/coco_swinl" \
 -c config/edpose.cfg.py \
 --options batch_size=4 epochs=60 lr_drop=55 num_body_points=17 backbone='swin_L_384_22k' \
 --dataset_file="sign" \
 --sign_path /D_data/SL/sign_data_processing/msasl_demo/images \
 --pretrain_model_path "../EDPose_pretrained/models/edpose_swinl_coco.pth" \
 --test \
 --kp_output_dir /D_data/SL/sign_data_processing/msasl_demo/keypoints_edpose_coco_swinl \
