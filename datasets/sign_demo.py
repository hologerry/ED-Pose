"""
COCO dataset which returns image_id for evaluation.
"""
import os

from pathlib import Path

import torch
import torch.utils.data
import torchvision.transforms.functional as F

from PIL import Image


__all__ = ["build"]


class SignData(torch.utils.data.Dataset):
    def __init__(self, root_path, image_set, transforms, return_masks):
        super().__init__()

        self.img_folder = root_path
        self.image_names = sorted(os.listdir(self.img_folder))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_file_name = self.image_names[idx]
        image_id = image_file_name.split(".")[0]

        target = {"image_id": image_id}
        org_img = Image.open(os.path.join(self.img_folder, image_file_name)).convert("RGB")
        img = Image.new("RGB", (org_img.width, 3 * org_img.height), (0, 0, 0))
        img.paste(org_img, (0, 0))
        w, h = img.size
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["boxes"] = torch.as_tensor([[0, 0, w, h]])
        img = F.to_tensor(img)
        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return img, target


def build(image_set, args):
    root = Path(args.sign_path)
    dataset = SignData(
        root,
        image_set,
        transforms=None,
        return_masks=args.masks,
    )

    return dataset
