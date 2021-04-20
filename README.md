# A Collection of GNN-based Fake News Detection Models
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://codeocean.com/capsule/3568017/tree) [![Build Status](https://travis-ci.com/safe-graph/DGFraud.svg?branch=master)](https://travis-ci.com/safe-graph/DGFraud) ![PyPI](https://img.shields.io/pypi/v/torch-geometric)

[Installation](#installation) | [Datasets](#datasets) | [User Guide](#user-guide) | [Leader Board](#leader-board) | [How to Contribute](#how-to-contribute)


This repo includes the implementation of our models and all baselines. The congfiguration can be setup with model arguments for each model. Note that the "hand" feature in model arguments represents the "Profile" feature.

We welcome contributions on adding new fraud detectors and extending the features of the toolbox. Some of the planned features are listed in [TODO list](#todo-list). 

If you use the code in your project, please cite the following paper:

SIGIR'21 ([PDF](https://arxiv.org/pdf/2005.00625.pdf))
```bibtex
@inproceedings{dou2021user,
  title={User Preference-aware Fake News Detection},
  author={Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao},
  booktitle={Proceedings of the 44nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```


## Installation

To run the code in this repo, you need to have `Python>=3.6`, `PyTorch>=1.6`, and `PyTorch-Geometric>=1.6.1`. Please follow the installation instructions of [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to install PyG.

Other packages can be installed using following commands:

```bash
git clone https://github.com/safe-graph/GNN-FakeNews.git
cd GNN-FakeNews
pip install -r requirements.txt
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

### Politifact
| Model  | Feature  | Accuracy  | F1  |
|-------|--------|--------|--------|
| **GCN** |  |  |  |
| **GAT** |  |  |  |
| **SAGE** |    |  |  |
| **GIN** |   |  |   |
| **GCNFN** |   |  |   |
| **BiGCN** |    |  | |
| **GNN-CL** |    |    |  |

### Gossipcop
| Model  | Feature  | Accuracy  | F1  |
|-------|--------|--------|--------|
| **GCN** |  |  |  |
| **GAT** |  |  |  |
| **SAGE** |    |  |  |
| **GIN** |   |  |   |
| **GCNFN** |   |  |   |
| **BiGCN** |    |  | |
| **GNN-CL** |    |    |  |



## How to Contribute
You are welcomed to submit your model, hyper-parameters, and results to this repo via create a pull request. After verifying the results, your model will be added to the benchmark.


