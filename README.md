# GraphCF
The code for paper "GraphCF: Drug-target interaction prediction via multi-feature fusion with contrastive graph neural network"


## 1. Overview

The repository is organized as follows:

+ `data/` contains the datasets used in the paper;
+ `parms_setting.py` contains hyperparameters adopted by GraphCF;
+ `data_preprocess.py` is the preprocess of data before training;
+ `layer.py` contains mix-hop GNN layers and contrastive GNN layers;
+ `instantiation.py` instantiates the GraphCF;
+ `train.py` contains the training and testing code on datasets;
+ `main.py` contains entry to GraphCF (e.g., normalize...);
+ The preprocessed 3D structure information files are available at -> https://www.scidb.cn/en/anonymous/bmVRdml1

## 2. Dependencies
* numpy == 1.23.5
* pandas == 2.0.3
* scikit-learn == 1.3.2
* scipy == 1.9.3
* torch == 1.8.0
* torch-geometric == 1.7.2
* networkx == 2.4


## 3. Example
To run GraphCF with following command:

```shell
python main.py --in_file human --out_file test.txt
```

