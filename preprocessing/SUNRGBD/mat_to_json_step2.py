
from mat4py import loadmat
import os
import numpy as np
from tqdm import tqdm
import json 

input_file = "/home/rayguan/Downloads/indoor_data/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
root = "/home/rayguan/multimodal/DATASETS/SUNRGBD"

out_file = os.path.join(root, "anno.json")

scene_lst = np.loadtxt(os.path.join(root, "scene_lst.txt"), dtype=str)
obj_lst = np.loadtxt(os.path.join(root, "obj_lst.txt"), dtype=str)

id_to_obj = {}
obj_to_id = {}
id_to_scene = {}
scene_to_id = {}


for i, o in enumerate(obj_lst):
    id_to_obj[i] = o
    obj_to_id[o] = i

pre = len(obj_lst)

for i, s in enumerate(scene_lst):
    id_to_scene[i + pre] = s
    scene_to_id[s] = i + pre
    

data = loadmat(input_file)

# image_lst = data["SUNRGBDMeta2DBB"]["rgbpath"]

image_lst = []

image_local_name_lst = data["SUNRGBDMeta2DBB"]["rgbname"]

image_dir_lst = data["SUNRGBDMeta2DBB"]["sequenceName"]

image_box_lst = data["SUNRGBDMeta2DBB"]["groundtruth2DBB"]

image_anno = []

for i, (img_dir, box_lst) in enumerate(zip(image_dir_lst, image_box_lst)):
    
    image_lst.append(os.path.join(img_dir, "image", image_local_name_lst[i]))
    
    path = os.path.join(root, img_dir, "scene.txt")
    with open(path, "r") as f:
        # scene = f.readlines()[0]
        scene = f.read().strip()
    
    ids = []
    
    if "classname" in box_lst:
        if isinstance(box_lst["classname"], list):
            for c in box_lst["classname"]:
                ids.append(obj_to_id[c.strip()])
        elif isinstance(box_lst["classname"], str):
            ids.append(obj_to_id[box_lst["classname"].strip()])

    ids.append(scene_to_id[scene])

    ids.sort()

    image_anno.append(ids)

out_dict = {
    "id_to_obj":id_to_obj,
    "id_to_scene": id_to_scene,
    "image_lst": image_lst,
    "image_anno": image_anno
}

out_f = open(out_file, "w")

json.dump(out_dict, out_f, indent=4)

# structure

# {
#     "id_to_obj": {id: object-name}
#     "id_to_scene": {id: scene-name}
#     "image_lst": [img_path]
#     "image_anno": [ ids ]
# }


