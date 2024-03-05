import os
from patch_util import read_patches

in_path = "./data/patches_for_static/"
out_path = "./out"

for patch in read_patches(in_path):
    if not os.path.exists(f"{out_path}/{patch.project}/{patch.name}"):
        os.makedirs(f"{out_path}/{patch.project}/{patch.name}")

    with open(f"{out_path}/{patch.project}/{patch.name}/{patch.project}_{patch.name}_s.java", "w") as s_file:
        s_file.writelines(patch.source)
    with open(f"{out_path}/{patch.project}/{patch.name}/{patch.project}_{patch.name}_t.java", "w") as t_file:
        t_file.writelines(patch.target)

