import os
import numpy as np

out_dir = "/home/rayguan/multimodal/DATASETS/simulation"

keyword_lst = [
    "BeanBag", # 0
    "BunkerBed", # 1
    "Bookcase", # 2
    "Dance", # 3
    "Idle", # 4
	"Clock", # 5
    "CoatRack", # 6
    "Crib", # 7
    "Curtain", # 8
    "Window", # 9
    "Suitcase", # 10
    "DishWasher", # 11
    "Oven", # 12
    "Fridge", # 13
    "Television", # 14
    "FramePicture", # 15
    "Stool", # 16
    "Trophy", # 17 
    "Mannequin", # 18
    "Planter", # 19
    # "", # 20
    # "", # 21
]
    

prompt_lst = [
    ["a beanbag"], # 0
    ["a bed"], # 1
    ["a book case"], # 2
    ["a guy dancing"], # 3
    ["a woman standing"], # 4
    ["a black clock"], # 5
    ["a coat rack"], # 6
    ["a crib"], # 7
    ["curtains"], # 8
    ["a window"], # 9
    ["a suitcase"], # 10
    ["a dish washer"], # 11
    ["an oven"], # 12
    ["a fridge"], # 13
    ["a television"], # 14
    ["a picture frame"], # 15
    ["a stool"], # 16
    ["a trophy"], # 17
    ["a mannequin"], # 18
    ["a planter"], # 19
    # [""], # 20
    # [""], # 21
]

prompt_lst = [l[0] for l in prompt_lst]

np.savetxt(os.path.join(out_dir, "prompt_lst.txt"), np.array(prompt_lst), fmt='%s')
np.savetxt(os.path.join(out_dir, "obj_lst.txt"), np.array(keyword_lst), fmt='%s')
