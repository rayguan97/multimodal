# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

import torch
from flava.data.transforms import (
    default_image_pretraining_transforms,
    default_text_transform,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmultimodal.models.flava.model import flava_model
from torchvision.datasets import CocoCaptions
from tqdm import tqdm 
import os 
from torchvision import transforms, utils
import pandas as pd
import json
from skimage import io, transform
from PIL import Image
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class SimDataset(Dataset):
    
    def __init__(self, root_dir, anno_name, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.anno_name = anno_name
        with open(os.path.join(root_dir, anno_name), "r") as f:
            self.anno = json.load(f)
        
        self.img_lst = self.anno["image_lst"]
        self.img_anno = self.anno["image_anno"]
        self.id_dict = self.anno["id_to_obj"]
        if "id_to_scene" in self.anno:
            self.id_dict = {**self.anno["id_to_obj"], **self.anno["id_to_scene"]}
            
        
        self.transform = transform

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):


        img_name = os.path.join(self.root_dir,
                                self.img_lst[idx])
        
        
        # image = io.imread(img_name)
        image = Image.open(img_name)
        text = ",".join([self.id_dict[str(i)] for i in self.img_anno[idx]])
        
        if self.transform:
            image, text = self.transform(image, text)

        return image, text



def compute_recall_relax(similarity_scores: torch.Tensor, targets, k: int = 5):
    dataset_size = similarity_scores.size(0)
    # targets = torch.arange(dataset_size).view(dataset_size, -1)
    targets = torch.tensor(targets).view(dataset_size, -1)
    _, topk_idx = torch.topk(similarity_scores, k)
    recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall

def compute_recall_relax_multi(similarity_scores: torch.Tensor, targets, k: int = 5):
    dataset_size = similarity_scores.size(0)
    # targets = torch.arange(dataset_size).view(dataset_size, -1)
    # targets = torch.tensor(targets).view(dataset_size, -1)
    # from IPython import embed;embed()
    _, topk_idx = torch.topk(similarity_scores, k)
    
    recall = 0
    for i, row in enumerate(topk_idx):
        for j in row:
            if j in targets[i]:
                recall += 1
                break
    # from IPython import embed;embed()

    # recall = targets.eq(topk_idx).sum()
    recall = recall / dataset_size
    return recall

def transform(image, target):
    _, image_transform = default_image_pretraining_transforms()
    transformed_image = image_transform(image)
    # Take the first caption for now
    transformed_text = default_text_transform()(target[0])
    return transformed_image, transformed_text

def transform2(image, target):
    _, image_transform = default_image_pretraining_transforms()
    transformed_image = image_transform(image)
    # Take the first caption for now
    transformed_text = default_text_transform()(target)
    # from IPython import embed;embed()

    return transformed_image, transformed_text

def collator(batch):
    texts = []
    images = torch.stack([x[0]["image"] for x in batch], dim=0)
    texts = torch.cat([torch.LongTensor(x[1]["input_ids"]) for x in batch], dim=0)
    return images, texts


def setup_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_root", help="Path to data root directory", default="/scratch0/rayguan/object_localization/GLIP/DATASET/coco/train2014")
    # parser.add_argument("--annotations", help="Path to annotation file", default="/scratch0/rayguan/object_localization/GLIP/DATASET/coco/annotations/captions_train2014.json")

    parser.add_argument("--data_root", help="Path to data root directory", default="/home/rayguan/multimodal/DATASETS/simulation")
    parser.add_argument("--annotations", help="Path to annotation file", default="anno.json")
    parser.add_argument("--batch_size", type=int, default=4)
    # parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def main():
    args = setup_args()
    # dataset = CocoCaptions(
    #     root=args.data_root, annFile=args.annotations, transforms=transform
    # )
    
    dataset = SimDataset(root_dir=args.data_root, anno_name=args.annotations, transform=transform2)

    flava = flava_model(pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    flava = flava.to(device)
    flava.eval()
    text_embeds = []
    image_embeds = []
    dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collator)

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        
        # from IPython import embed;embed()
        # logger.info(f"Batch id {batch_idx}")
        image, text = batch
        _, text_emb = flava.encode_text(text.to(device), projection=True)
        _, image_emb = flava.encode_image(image.to(device), projection=True)
        text_embeds.append(text_emb.detach().cpu())
        image_embeds.append(image_emb.detach().cpu())
        
        # if batch_idx > 5:
        #     break
        

    image_embeds = torch.cat(image_embeds, 0)
    text_embeds = torch.cat(text_embeds, 0)

    # from IPython import embed;embed()
    
    # text_unique, idx = np.unique(text_embeds, return_index=True, axis=0)

    text_unique, idx, idx_rev = np.unique(text_embeds, return_index=True, return_inverse=True, axis=0)
    
    anno_lst = defaultdict(list)
    
    for i, id in enumerate(idx_rev):
        anno_lst[id].append(i)

    # from IPython import embed;embed()

    image_embeds = nn.functional.normalize(image_embeds, dim=-1)
    text_embeds = nn.functional.normalize(text_embeds[idx], dim=-1)

    similarity_scores = image_embeds @ text_embeds.t()
    similarity_scores_t = similarity_scores.t()

    image_to_text_relax_r1 = compute_recall_relax(similarity_scores, idx_rev, k=1)
    image_to_text_relax_r5 = compute_recall_relax(similarity_scores, idx_rev, k=5)
    text_to_image_relax_r1 = compute_recall_relax_multi(similarity_scores_t, anno_lst, k=1)
    text_to_image_relax_r5 = compute_recall_relax_multi(similarity_scores_t, anno_lst, k=5)

    logger.info(f"image_to_text_recall@1 {image_to_text_relax_r1}")
    logger.info(f"image_to_text_recall@5 {image_to_text_relax_r5}")
    logger.info(f"text_to_image_recall@1 {text_to_image_relax_r1}")
    logger.info(f"text_to_image_recall@5 {text_to_image_relax_r5}")


if __name__ == "__main__":
    main()
