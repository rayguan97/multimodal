import os
import numpy as np
from tqdm import tqdm
import json

root = "/home/rayguan/multimodal/DATASETS/simulation"
img_label_path = os.path.join(root, "img_label.json")

out_file = os.path.join(root, "anno.json")
obj_lst = np.loadtxt(os.path.join(root, "obj_lst.txt"), dtype=str)

id_to_obj = {}
obj_to_id = {}

for i, o in enumerate(obj_lst):
    id_to_obj[i] = o
    obj_to_id[o] = i

img_lst = np.loadtxt(os.path.join(root, "img_lst.txt"), dtype=str)


with open(img_label_path, "r") as f:
    img_label = json.load(f)



image_anno = []

for i in img_lst:
    ids = []
    
    for k in img_label.keys():
        if i in img_label[k]:
            ids.append(int(k.split("_")[1]))
    
    image_anno.append(ids)


img_lst = [os.path.join("images", i) for i in img_lst]

out_dict = {
    "id_to_obj":id_to_obj,
    "image_lst": img_lst,
    "image_anno": image_anno
}

out_f = open(out_file, "w")

json.dump(out_dict, out_f, indent=4)
