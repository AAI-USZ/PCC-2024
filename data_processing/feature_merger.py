
ASE_features = []
with open(f"../data/merged_static_features.csv") as csv_file:
    ASE_features = csv_file.readlines()

sim_features = []
with open(f"../data/sim_features.csv") as csv_file:
    sim_features = csv_file.readlines()

ODS_features = []
with open(f"../data/ODS_features.csv") as csv_file:
    ODS_features = csv_file.readlines()

n = 0
with open(f"../data/ALL_features.csv", "w") as csv_file:
    csv_file.write(f"{ASE_features[0][:-1]},{sim_features[0][:-2]},{ODS_features[0]}")
    for ASE_line in ASE_features[1:]:
        columns = ASE_line.split(",")
        split = columns[0].split(".")

        tool = split[8]
        project = split[9]
        id = split[10]
        mutant_id = 1
        if len(split) > 11:
            mutant_id = split[11]

        sim_matching = [s for s in sim_features if f"{project}_{tool}_{id}_{mutant_id}" in s]
        ODS_matching = [s for s in ODS_features if f"patch{mutant_id}-{project}-{id}-{tool}" in s]

        if len(sim_matching) == 0:
            print(f"No match for {project}_{tool}_{id}_{mutant_id}")
            continue
        if len(ODS_matching) == 0:
            print(f"No match for {mutant_id}-{project}-{id}-{tool}")
            continue


        csv_file.write(f"{ASE_line[:-1]},{sim_matching[0][:-2]},{ODS_matching[0]}")
        n += 1
        #print(f"Merged {n} entities so far.")

