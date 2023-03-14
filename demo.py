import argparse
import json
import os
import random
import sys
import time

from pathlib import Path

import numpy as np
import torch

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, DistributedSampler

import util.misc as utils

from datasets import build_dataset
from util.config import Config, DictAction
from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.utils import ModelEma, to_device


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config_file", "-c", type=str, required=True)
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", type=str, default=None)
    parser.add_argument("--sign_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default="MSASL")
    parser.add_argument("--split", type=str, default="train")

    parser.add_argument("--remove_difficult", action="store_true")
    parser.add_argument("--fix_size", action="store_true")

    # training parameters
    parser.add_argument("--output_dir", default="", help="path where to save, empty for no saving")
    parser.add_argument("--kp_output_dir", default="", help="path where to save, empty for no saving")

    parser.add_argument("--note", default="", help="add some notes to the experiment")
    parser.add_argument("--device", default="cuda", help="device to use for training / testing")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--pretrain_model_path", help="load from other checkpoint")
    parser.add_argument("--finetune_ignore", type=str, nargs="+")
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--find_unused_params", action="store_true")

    parser.add_argument("--save_results", action="store_true")
    parser.add_argument("--save_log", action="store_true")

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--rank", default=0, type=int, help="number of distributed processes")
    parser.add_argument("--local_rank", type=int, help="local rank for DistributedDataParallel")
    parser.add_argument("--amp", action="store_true", help="Train with mixed precision")

    return parser


def build_model_main(args):
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def main(args):
    utils.init_distributed_mode(args)
    # print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = Config.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, "use_ema", None):
        args.use_ema = False
    if not getattr(args, "debug", None):
        args.debug = False
    if args.dataset_file == "sign":
        assert os.path.exists(args.sign_path)

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(
        output=os.path.join(args.output_dir, "info.txt"), distributed_rank=args.rank, color=False, name="detr"
    )
    # logger.info("git:\n  {}\n".format(utils.get_sha()))
    # logger.info("Command: " + " ".join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        # print("args:", vars(args))
        with open(save_json_path, "w") as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    # logger.info("world size: {}".format(args.world_size))
    # logger.info("rank: {}".format(args.rank))
    # logger.info("local_rank: {}".format(args.local_rank))
    # logger.info("args: " + str(args) + "\n")

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    # print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # logger.info("number of params:" + str(n_parameters))
    # logger.info(
    #     "params:\n" + json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2)
    # )

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)

    dataset_test = build_dataset(image_set="test", args=args)

    if args.distributed:
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_test = DataLoader(
        dataset_test,
        args.batch_size,
        sampler=sampler_test,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
        from collections import OrderedDict

        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True

        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict(
            {k: v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)}
        )

        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))

        if args.use_ema:
            if "ema_model" in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint["ema_model"]))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)

    if args.test:
        os.environ["TEST_FLAG"] = "TRUE"
        test(
            model,
            None,
            postprocessors,
            data_loader_test,
            None,
            device,
            args.output_dir,
            wo_class_error=wo_class_error,
            args=args,
        )

        return


@torch.no_grad()
def test(
    model,
    criterion,
    postprocessors,
    data_loader,
    base_ds,
    device,
    output_dir,
    wo_class_error=False,
    args=None,
    logger=None,
):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False
    model.eval()
    # criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter("class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}"))
    header = "Test:"
    iou_types = tuple(k for k in ("bbox", "keypoints"))
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))

    _cnt = 0
    os.makedirs(args.kp_output_dir, exist_ok=True)

    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors["bbox"](outputs, orig_target_sizes)
        # print("type samples ", type(samples))
        images = samples.to_img_list()
        # print("type images 0 ", type(images[0]))
        # print("shape images 0", images[0].shape)
        # print("results len ", len(results))
        # print("targets len ", len(targets))
        for idx, res in enumerate(results):

            # print(type(res))
            # print(res.keys())

            best_idx = res["scores"].argmax()
            # box = res["boxes"][best_idx].cpu().numpy()
            # score = res["scores"][best_idx].cpu().numpy()
            # label = res["labels"][best_idx].cpu().numpy()
            keypoints = res["keypoints"][best_idx].cpu().numpy()
            image_id = targets[idx]["image_id"]

            if "video_id" in targets[idx]:
                video_id = targets[idx]["video_id"]
                os.makedirs(os.path.join(args.kp_output_dir, video_id), exist_ok=True)

            kp_dict = {"keypoints": keypoints.tolist()}

            output_json_path = os.path.join(args.kp_output_dir, f"{image_id}.json")
            with open(output_json_path, "w") as f:
                json.dump(kp_dict, f)

        _cnt += 1
        # if args.debug:
        #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
