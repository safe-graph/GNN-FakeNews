# Benchmarking GNN-based Fake News Detection
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/3568017/tree) [![Build Status](https://travis-ci.com/safe-graph/DGFraud.svg?branch=master)](https://travis-ci.com/safe-graph/DGFraud) ![PyPI](https://img.shields.io/pypi/v/torch-geometric)


**Introduction** 

This repo includes the implementation of our models and all baselines. The congfiguration can be setup with model arguments for each model. Note that the "hand" feature in model arguments represents the "Profile" feature.

We welcome contributions on adding new fraud detectors and extending the features of the toolbox. Some of the planned features are listed in [TODO list](#todo-list). 

If you use the code in your project, please cite the following paper:

SIGIR'21 ([PDF](https://arxiv.org/pdf/2005.00625.pdf))
```bibtex
@inproceedings{liu2020alleviating,
  title={User Preference-aware Fake News Detection},
  author={Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao},
  booktitle={Proceedings of the 44nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

**Table of Contents**
- [Installation](#installation)
- [Datasets](#datasets)
- [User Guide](#user-guide)
- [Leader Board](#leader-board)
- [How to Contribute](#how-to-contribute)


CSI and SAFE are implemented in ```baselines.py```

GNN-CL is implemented in ```gnncl.py```

GCNFN and UPFD-GCNFN are implemented in ```gcnfn.py```

UPFD-GCN and UPFD-SAGE are implemented in ```gnn.py```


## Installation
```bash
git clone https://github.com/safe-graph/DGFraud.git
cd DGFraud
python setup.py install
```
### Requirements
```bash
* Python>=3.6
* PyTorch>=1.6
* PyTorch_Geometric>=1.6.1
* keras>=2.2.4
* scikit-learn>=0.22.1
* tqdm>=4.31.1
* numpy>=1.19.4
* scipy>=1.5.2
```

## Datasets

You can download the dataset via the link below:

https://mega.nz/file/jwggTC4D#MT1cOOzAOOjMU8i_kBTJ07vO-jQUW6Tu_YKY_ON97h4

## User Guide

### Running the example code
You can find the implemented models in `algorithms` directory. For example, you can run Player2Vec using:
```bash
python Player2Vec_main.py 
```
You can specify parameters for models when running the code.

### Running on your datasets
Have a look at the load_data_dblp() function in utils/utils.py for an example.

In order to use your own data, you have to provide:
* adjacency matrices or adjlists (for GAS);
* a feature matrix
* a label matrix
then split feature matrix and label matrix into testing data and training data.

You can specify a dataset as follows:
```bash
python xx_main.py --dataset your_dataset 
```
or by editing xx_main.py

### The structure of code
The repository is organized as follows:
- `algorithms/` contains the implemented models and the corresponding example code;
- `base_models/` contains the basic models (GCN);
- `dataset/` contains the necessary dataset files;
- `utils/` contains:
    * loading and splitting the data (`data_loader.py`);
    * contains various utilities (`utils.py`).


## Leader Board


## Model Comparison
| Model  | Application  | Graph Type  | Base Model  |
|-------|--------|--------|--------|
| **SemiGNN** | Financial Fraud  | Heterogeneous   | GAT, LINE, DeepWalk |
| **Player2Vec** | Cyber Criminal  | Heterogeneous | GAT, GCN|
| **GAS** | Opinion Fraud  | Heterogeneous | GCN, GAT |
| **FdGars** |  Opinion Fraud | Homogeneous | GCN |
| **GeniePath** | Financial Fraud | Homogeneous | GAT  |
| **GEM** | Financial Fraud  | Heterogeneous |GCN |
| **GraphSAGE** | Opinion Fraud  | Homogeneous   | GraphSAGE |
| **GraphConsis** | Opinion Fraud  | Heterogeneous   | GraphSAGE |
| **HACUD** | Financial Fraud | Heterogeneous | GAT |


## How to Contribute
You are welcomed to contribute to this open-source toolbox. The detailed instructions will be released soon. Currently, you can create issues or send email to [bdscsafegraph@gmail.com](mailto:bdscsafegraph@gmail.com) for inquiry.



The hyperparameters are reported in the following table.

### Politifact
| model   | feature | epoch  | lr | emb_size | batch_num  |
|-------|--------|--------|--------|-----------|-----------|
| CSI  | news text  | 300  | 0.01 | 128 | 128  |
| SAFE  | news text  | 300  | 0.01 | 128 | 128  |
| GNN-CL  | Profile  | 60  | 0.001 | 128 | 128  |
| GCNFN  | content  | 50  | 0.001 | 128 | 128  |
| UPFD-GCN  | GloVe, Profile  | 60  | 0.001 | 128 | 128  |
| UPFD-GCN  | BERT  | 100  | 0.001 | 128 | 128  |
| UPFD-SAGE  | GloVe, Profile  | 100  | 0.001 | 128 | 128  |
| UPFD-SAGE  | BERT  | 60  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | Profile  | 80  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | GloVe  | 60  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | BERT  | 45  | 0.001 | 128 | 128  |

### Gossipcop
| model   | feature | epoch  | lr | emb_size | batch_num  |
|-------|--------|--------|--------|-----------|-----------|
| CSI  | news text  | 150  | 0.01 | 128 | 128  |
| SAFE  | news text  | 150  | 0.01 | 128 | 128  |
| GNN-CL  | Profile  | 40  | 0.001 | 128 | 128  |
| GCNFN  | content  | 50  | 0.001 | 128 | 128  |
| UPFD-GCN  |Profile  | 50  | 0.01 | 128 | 128  |
| UPFD-GCN  | GloVe, BERT  | 50  | 0.001 | 128 | 128  |
| UPFD-SAGE  | Profile  | 50  | 0.01 | 128 | 128  |
| UPFD-SAGE  | GloVe BERT  | 80  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | Profile, GloVe, BERT  | 50  | 0.001 | 128 | 128  |
