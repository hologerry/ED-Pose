import os
import pickle

from pathlib import Path

import torch
import torch.utils.data
import torchvision.transforms.functional as F

from PIL import Image


__all__ = ["build"]


class SignVideoData(torch.utils.data.Dataset):
    def __init__(self, root_path, dataset_name, split, transforms, return_masks):
        super().__init__()

        split_fn = os.path.join(root_path, dataset_name, f"{split}.pkl")
        with open(split_fn, "rb") as f:
            self.split_dicts = pickle.load(f)

        self.frames_dir = os.path.join(root_path, dataset_name, "frames")
        self.frames_names = []
        for vid_dict in self.split_dicts:
            vid_name = vid_dict["name"]
            seq_len = vid_dict["seq_len"]
            for frame_i in range(seq_len):
                self.frames_names.append(f"{vid_name}/{frame_i:04d}.png")

    def __len__(self):
        return len(self.frames_names)

    def __getitem__(self, idx):
        image_file_name = self.frames_names[idx]
        image_id = image_file_name.split(".")[0]

        target = {"image_id": image_id, "video_id": image_id.split("/")[0]}
        org_img = Image.open(os.path.join(self.frames_dir, image_file_name)).convert("RGB")
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
    dataset = SignVideoData(
        root,
        dataset_name=args.dataset_name,
        split=args.split,
        transforms=None,
        return_masks=args.masks,
    )

    return dataset
