import torch.utils.data
import torchvision

from .coco import build as build_coco
from .sign_demo import build as build_sign
from .sign_video import build as build_sign_video


# from .crowdpose import build as build_crowdpose


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == "coco":
        return build_coco(image_set, args)
    elif args.dataset_file == "sign":
        return build_sign(image_set, args)
    elif args.dataset_file == "sign_video":
        return build_sign_video(image_set, args)
    # if args.dataset_file == "crowdpose":
    #     return build_crowdpose(image_set, args)
    raise ValueError(f"dataset {args.dataset_file} not supported")
