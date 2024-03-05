import os


class Patch:
    def __init__(self, project, name, source="", target=""):
        self.project = project
        self.name = name
        self.source = source
        self.target = target


def read_patches(in_path):
    patches = []
    process_index = 0

    for patch_dir in ["Patches_ICSE", "Patches_others"]:
        for state_dir in ["Dcorrect", "Doverfitting"]:
            for _, tools, _ in os.walk(f"{in_path}/{patch_dir}/{state_dir}/"):
                for tool in tools:
                    for project_root, projects, _ in os.walk(f"{in_path}/{patch_dir}/{state_dir}/{tool}/"):
                        for project in projects:
                            for index_root, indices, _ in os.walk(f"{project_root}/{project}/"):
                                for index in indices:
                                    for buggy_index in range(1, 50):
                                        patch = Patch(f"{project}_{tool}_{index}_{buggy_index}", "ClassName")
                                        file_name = ""
                                        if "others" in project_root:
                                            file_name = f"{in_path}/{patch_dir}/{state_dir}/{tool}/{project}/{index}/{buggy_index}/buggy1.java"
                                        else:
                                            file_name = f"{in_path}/{patch_dir}/{state_dir}/{tool}/{project}/{index}/buggy{buggy_index}.java"
                                        if os.path.isfile(file_name):
                                            with open(file_name) as s_file:
                                                patch.source = s_file.readlines()
                                            with open(file_name.replace("buggy", "tool-patch")) as t_file:
                                                patch.target = t_file.readlines()
                                            patches.append(patch)
                                            process_index += 1
                                            print(f"Found and saved: {patch.project} ({process_index}/903)")
    return patches