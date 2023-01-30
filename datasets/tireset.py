from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from datasets.coco import ConvertCocoPolysToMask
from PIL import Image
import datasets.transforms as T


class TireCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(TireCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def _load_iamge(self,id:int)-> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(path).convert("RGB")

    def preload(self,id):
        id = self.ids[id]
        image = self._load_image(id)
        target = self._load_target(id)
        return image, target
    
    def __getitem__(self, idx):
        # img, target = super(TireCocoDetection, self).__getitem__(idx)
        img,target = self.preload(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def build_tire(image_set, args):
    coco_root = Path(args.coco_path)
    img_root = Path(args.image_root)
    assert coco_root.exists(), f'provided COCO path {coco_root} does not exist'
    assert img_root.exists(), f'provided COCO path {img_root} does not exist'
    
    mode = 'instances'
    PATHS = {
        "train": (img_root, coco_root / f'train.json'),
        "val": (img_root, coco_root / f'val.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = TireCocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')
