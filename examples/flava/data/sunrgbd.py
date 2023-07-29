
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
import os
import json
# import transforms
from PIL import Image

# from .transforms import (
#     default_image_pretraining_transforms,
#     default_text_transform,
#     default_torchvision_transforms,
#     encode_text_batch,
#     pad_batch,
#     TEXT_DEFAULT_TOKENIZER,
#     TEXT_WHOLE_WORD_MASK_TOKENIZER,
#     VL_MAX_LENGTH_DEFAULT,
#     VLTransform,
# )


class SUNRGBD(Dataset):


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
        
        self.img_lst_tmp = self.anno["image_lst"]
        self.img_anno_tmp = self.anno["image_anno"]
        self.id_dict = self.anno["id_to_obj"]
        if "id_to_scene" in self.anno:
            self.id_dict = {**self.anno["id_to_obj"], **self.anno["id_to_scene"]}

        self.img_lst = []
        self.img_anno = []

        for i in range(len(self.img_lst_tmp)):
            for idx in self.img_anno_tmp[i]:
                self.img_lst.append(self.img_lst_tmp[i])
                self.img_anno.append(idx)
            
        
        self.transform = transform

    def __len__(self):
        return len(self.img_lst)

    def __getitem__(self, idx):


        img_name = os.path.join(self.root_dir,
                                self.img_lst[idx])
        
        
        # image = io.imread(img_name)
        image = Image.open(img_name)
        # text = ",".join([self.id_dict[str(i)] for i in self.img_anno[idx]])
        text = self.id_dict[str(self.img_anno[idx])]
        out_dict = {"image": image, "text": text}
        if self.transform:
            out_dict = self.transform(out_dict)

        return out_dict

    def set_transform(self, transform):
        self.transform = transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", help="Path to data root directory", default="/home/rayguan/multimodal/DATASETS/SUNRGBD")
    parser.add_argument("--annotations", help="Path to annotation file", default="anno.json")
    parser.add_argument("--batch_size", type=int, default=4)

    args = parser.parse_args()

    # transform = VLTransform
    transform = None

    dataset = SUNRGBD(root_dir=args.data_root, anno_name=args.annotations, transform=transform)

    d = DataLoader(
                dataset,
                shuffle=True,
                batch_size=4,
                num_workers=4,
            )
