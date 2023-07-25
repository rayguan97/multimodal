from mat4py import loadmat
import os
import numpy as np
from tqdm import tqdm

root = "/home/rayguan/multimodal/DATASETS/SUNRGBD"
input_file = "/home/rayguan/Downloads/indoor_data/SUNRGBD/SUNRGBDMeta2DBB_v2.mat"
out_dir = "/home/rayguan/multimodal/DATASETS/SUNRGBD"


scene_set = set()
obj_set = set()

data = loadmat(input_file)



image_dir_lst = data["SUNRGBDMeta2DBB"]["sequenceName"]

# image_name_lst = data["SUNRGBDMeta2DBB"]["rgbpath"]

for image_dir in tqdm(image_dir_lst):
    path = os.path.join(root, image_dir, "scene.txt")
    with open(path, "r") as f:
        # scene = f.readlines()[0]
        scene = f.read()

    scene_set.add(scene.strip())    
    # from IPython import embed;embed()


image_box_lst = data["SUNRGBDMeta2DBB"]["groundtruth2DBB"]

# from IPython import embed;embed()


for box_lst in tqdm(image_box_lst):
    if "classname" in box_lst:
        if isinstance(box_lst["classname"], list):
            for c in box_lst["classname"]:
                obj_set.add(c.strip())
        elif isinstance(box_lst["classname"], str):
            obj_set.add(box_lst["classname"].strip())
            
        
# from IPython import embed;embed()

np.savetxt(os.path.join(out_dir, "scene_lst.txt"), np.array(list(scene_set)), fmt='%s')
np.savetxt(os.path.join(out_dir, "obj_lst.txt"), np.array(list(obj_set)), fmt='%s')

