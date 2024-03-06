
capgen = []
ssfix = []

with open(f"../data/standalone/ssfix_metrics.csv") as csv_file:
    ssfix = csv_file.readlines()

with open(f"../data/standalone/capgens3_metrics.csv") as csv_file:
    capgen = csv_file.readlines()

with open(f"./data/Hand_crafted_features.csv", "w") as csv_file:
    csv_file.write("bug,s3-tool-buggy,s3-tool-developer,AST-tool-buggy,AST-tool-developer,Cosine-tool-buggy,Cosine-tool-developer,s3variable-tool-buggy,s3variable-tool-developer,capgen-tool,capgen-developer,variable-tool-buggy,variable-tool-developer,syntax-tool-buggy,syntax-tool-developer,semantic-tool-buggy,semantic-tool-developer,tool,project,id,mutant-id,line_in_buggy,line_in_patched,structural_score,conceptual_score,sum,correct\n")
    for line in capgen:
        columns = line.split(",")
        split = columns[0].split(".")

        tool = split[8]
        project = split[9]
        id = split[10]
        mutant_id = 0
        if len(split) > 11:
            mutant_id = split[11]

        matching = [s for s in ssfix if f"{tool},{project},{id},{mutant_id}" in s]
        csv_file.write(f"{line[:-2]},{matching[0]}")

