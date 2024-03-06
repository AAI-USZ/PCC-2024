# Patch Correctness Check

Due to the potential imperfections in Automated Program Repair (APR), generated patches might be incorrect. A significant portion of the literature on APR focuses on this issue, usually referred to as Patch Correctness Check (PCC). In the following repository and paper we provide a systematic overview in this domain.

## Feature Extraction, Learning and Selection in Support of Patch Correctness Assessment

This repository contains open science data used in the paper 

>  **Feature Extraction, Learning and Selection in Support of Patch Correctness Assessment**

submitted at the Proceedings of the [19th International Conference on Software Technologies (ICSOFT '24)](https://icsoft.scitevents.org). If you use this repository for academic purposes, please cite the appropriate publication:
```
@inproceedings{pcc,
 title = {Feature Extraction, Learning and Selection in Support of Patch Correctness Assessment},
 author = {Anonymous},
 booktitle={Proceedings of the 19th International Conference on Software Technologies - ICSOFT}
 year = {2024},
 doi = {},
}
```

In each directory we provide a README file that describes the structure of the folder in question.

### Data processing
Source code related to data processing can be found in the [data_processing](./data_processing) subfolder. We preprocessed data taken from a [recent study](https://github.com/claudeyj/patch_correctness/tree/master), extracted metrics using [ODS](https://github.com/ASSERT-KTH/ODSExperiment) and [Coming](https://github.com/SpoonLabs/coming). 

### Features and experimental data
All of the extracted features are stored in the [data](./data) folder. We publish data copied from previous works and newly extracted features as well. The original source files and patches are also available in the folder.

### Machine Learning
Source and utility files can be found under [utility](./utility) folder. 

### Experimental files
To see the conducted experiments, one can examine the files under [experiments](./experiments) folder. In the paper we trained and evaluated each model using 10 different random seeds and averaged the results on each.
