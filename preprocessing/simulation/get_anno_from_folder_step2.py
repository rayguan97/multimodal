
import numpy as np
import os
import json

anno_folder = "/scratch0/rayguan/object_localization/benchmark/label1"


root = "/home/rayguan/multimodal/DATASETS/simulation"
out_file = os.path.join(root, "img_label.json")

obj_lst = np.loadtxt(os.path.join(root, "obj_lst.txt"), dtype=str)

print(os.listdir(anno_folder))

anno_raw = {}

for d in os.listdir(anno_folder):
    c, i = d.split("_")
    assert obj_lst[int(i)] == c
        
    anno_raw[d] = os.listdir(os.path.join(anno_folder, d))


out_f = open(out_file, "w")

json.dump(anno_raw, out_f, indent=4)