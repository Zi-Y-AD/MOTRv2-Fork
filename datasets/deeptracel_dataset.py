import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
from pathlib import Path
import pdb
from torchvision import transforms as T
from util.misc import Instances


class MOTDataset(data.Dataset):
    # set the right path to the custom dataset in this line
    def __init__(self, root, transform=None, split="train"):
        """
        Args:
            root: chemin vers le dataset MOTChallenge custom
                  (ex: /data/MOT17/train/MOT17-02)
            transform: augmentations ou normalisation
            split: 'train' ou 'val'
        """
        self.root = root
        # pdb.set_trace()
        self.transform = transform or T.Compose([T.ToTensor()])
        self.split = split

        # Dossier images
        self.img_dir = os.path.join(root, "img1")
        self.img_files = sorted(os.listdir(self.img_dir))

        # Charger les annotations GT
        self.annotations = self._load_annotations()
        
    def set_epoch(self, epoch):
        # nécessaire pour DistributedSampler
        pass
    

    def _load_annotations(self):
        ann_path = os.path.join(self.root, "gt", "gt.txt")
        annotations = {}

        if not os.path.exists(ann_path):
            print(f"⚠️ Pas de fichier GT trouvé : {ann_path}")
            return annotations

        with open(ann_path, "r") as f:
            for line in f.readlines():
                frame, obj_id, x, y, w, h, conf, cls, vis = line.strip().split(",")
                frame, obj_id = int(frame), int(obj_id)
                x, y, w, h = float(x), float(y), float(w), float(h)

                # Convertir en [x1, y1, x2, y2]
                box = [x, y, x + w, y + h]

                if frame not in annotations:
                    annotations[frame] = []

                annotations[frame].append({
                    "id": obj_id,
                    "box": box,
                    "label": 1  # ex: "personne" par défaut
                })
        return annotations

    def __len__(self):
        return len(self.img_files)

#     def __getitem__(self, idx):
#         img_name = self.img_files[idx]
#         img_path = os.path.join(self.img_dir, img_name)

#         # Charger image
#         img = Image.open(img_path).convert("RGB")
#         img = self.transform(img)

#         # Frame id
#         frame_id = int(os.path.splitext(img_name)[0])

#         # Charger annotations
#         annos = self.annotations.get(frame_id, [])

#         boxes = torch.as_tensor([a["box"] for a in annos], dtype=torch.float32)
#         labels = torch.as_tensor([a["label"] for a in annos], dtype=torch.int64)
#         ids = torch.as_tensor([a["id"] for a in annos], dtype=torch.int64)

#         target = {
#             "boxes": boxes if isinstance(boxes, list) else boxes.cpu().numpy().tolist(),
#             "labels": labels if isinstance(labels, list) else labels.cpu().numpy().tolist(),
#             "image_id": ids if isinstance(ids, list) else ids.cpu().numpy().tolist(),
#         "orig_size": tuple([img.shape[1], img.shape[2]]), # (H, W)
#             "size": tuple([img.shape[1], img.shape[2]]) }      # (H, W) 

#         # if self.transform is not None:
#         #     img, target = self.transform(img, target)
        
#         sample = {
#         "img": img,   # torch.Tensor
#         "gt_instances": {
#             "boxes": boxes if isinstance(boxes, list) else boxes.cpu().numpy().tolist(),    # doit être list
#             "labels": labels if isinstance(labels, list) else labels.cpu().numpy().tolist(),  # doit être list
#         },
#         "image_id": [int(idx)],  # doit être list[int]
#         "orig_size": (int(img.shape[1]), int(img.shape[2])),
#         "size": (int(img.shape[1]), int(img.shape[2])),
#         }

#         print("DEBUG __getitem__", {k: type(v) for k,v in sample.items()})
#         print("DEBUG gt_instances", {k: type(v) for k,v in sample["gt_instances"].items()})

#         return sample



#         return {"img": img, "target": target}

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # Charger image
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Frame id
        frame_id = int(os.path.splitext(img_name)[0])

        # Charger annotations
        annos = self.annotations.get(frame_id, [])

        boxes = torch.as_tensor([a["box"] for a in annos], dtype=torch.float32)
        labels = torch.as_tensor([a["label"] for a in annos], dtype=torch.int64)
        obj_ids = torch.as_tensor([a["id"] for a in annos], dtype=torch.int64)

        # Créer un objet Instances compatible MOTR
        gt_instances = Instances()
        gt_instances.boxes = boxes
        gt_instances.labels = labels
        gt_instances.obj_ids = obj_ids  # MOTR attend obj_ids

        sample = {
            "img": img,  # torch.Tensor
            "gt_instances": gt_instances,  # Instances
            "image_id": [int(idx)],        # list[int]
            "orig_size": (int(img.shape[1]), int(img.shape[2])),
            "size": (int(img.shape[1]), int(img.shape[2])),
        }

        print("DEBUG __getitem__", {k: type(v) for k,v in sample.items()})
        print("DEBUG gt_instances", {k: type(getattr(gt_instances, k)) for k in ['boxes','labels','obj_ids']})

        return sample

