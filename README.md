# GNN-based Fake News Detection
[![Open in Code Ocean](https://codeocean.com/codeocean-assets/badge/open-in-code-ocean.svg)](https://github.com/safe-graph/GNN-FakeNews)
[![Scrutinizer Code Quality](https://scrutinizer-ci.com/g/safe-graph/GNN-FakeNews/badges/quality-score.png?b=main)](https://scrutinizer-ci.com/g/safe-graph/GNN-FakeNews/?branch=main)
[![LICENSE](https://img.shields.io/github/license/safe-graph/GNN-FakeNews)](https://github.com/safe-graph/GNN-FakeNews/blob/main/LICENSE)


[Installation](#installation) | [Datasets](#datasets) | [User Guide](#user-guide) | [Leader Board](#leader-board) | [How to Contribute](#how-to-contribute)


This repo includes the Pytorch-Geometric implementation of a series of Graph Neural Network (GNN) based fake news detection models.
All [GNN models](#leader-board) are implemented and evaluated under the User Preference-aware Fake News Detection (UPFD) framework.
The fake news detection problem is instantiated as a graph classification task under the UPFD framework. 


You can run the project on the virtual machine hosted by the Code Ocean without manual configuration.

We welcome contributions of results of existing models and the SOTA results of new models based on our dataset.
You can check the [Leader Board](#leader-board) for implemented models and their performances.

If you use the code in your project, please cite the following paper:

SIGIR'21 ([PDF](https://github.com/safe-graph/GNN-FakeNews))
```bibtex
@inproceedings{dou2021user,
  title={User Preference-aware Fake News Detection},
  author={Dou, Yingtong and Shu, Kai and Xia, Congying and Yu, Philip S. and Sun, Lichao},
  booktitle={Proceedings of the 44nd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2021}
}
```

## Installation

To run the code in this repo, you need to have `Python>=3.6`, `PyTorch>=1.6`, and `PyTorch-Geometric>=1.6.1`.
Please follow the installation instructions of [PyTorch-Geometric](https://github.com/rusty1s/pytorch_geometric) to install PyG.

Other dependencies can be installed using the following commands:

```bash
git clone https://github.com/safe-graph/GNN-FakeNews.git
cd GNN-FakeNews
pip install -r requirements.txt
```

## Datasets

The dataset can be loaded using the PyG API. You can download the dataset (2.66GB) via the link below and
unzip the data under the `\data` directory.

https://mega.nz/file/j5ZFEK7Z#KDnX2sjg65cqXsIRi0cVh6cvp7CDJZh1Zlm9-Xt28d4

The dataset includes fake&real news propagation networks on Twitter built according to fact-check information from
[Politifact](https://www.politifact.com/) and [Gossipcop](https://www.gossipcop.com/).
The news retweet graphs were originally extracted by [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
We crawled near 20 million historical tweets from users participated in fake news propagation in FakeNewsNet to
generate user features in the dataset.

The statistics of the dataset is shown below:

| Data  | #Graphs  | #Total Nodes  | #Total Edges  | #Avg. Nodes per Graph  |
|-------|--------|--------|--------|--------|
| Politifact | 314   |  41,054  | 40,740 |  131 |
| Gossipcop |  5464  |  314,262  | 308,798  |  58  |


Due to the Twitter policy, we could not release the crawled user historical tweets publicly.
To get the corresponding Twitter user information, you can refer to news list under `\data`
and map the news id to [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet).
Then, you can get the Twitter user id and their information following instructions on FakeNewsNet.
In this project, we use [Tweepy](https://www.tweepy.org/) and [Twitter Developer API](https://developer.twitter.com/en) to get the user information.

We incorporate four feature types in the dataset, the 768-dimensional `bert` and 300-dimensional `spacy` features 
are encoded using pretrained [BERT](https://github.com/hanxiao/bert-as-service) and [spaCy](https://spacy.io/models/en#en_core_web_lg) word2vec, respectively.
The 10-dimensional `profile` feature is obtained from a Twitter account's profile.
You can refer to [profile_feature.py](https://github.com/safe-graph/GNN-FakeNews/blob/master/utils/profile_feature.py) for profile feature extraction.
The 310-dimensional `content` feature is composed of 300-dimensional user comment word2vec (spaCy) embeddings
plus 10-dimensional profile features.

The graph data is a tree-structured social context graph where the root node represents the news,
the leaf nodes are Twitter users who retweeted the root news.
The following figure shows the UPFD framework including the dataset construction details 
You can refer to the [paper](https://arxiv.org/pdf/2005.00625.pdf) for more details about the dataset.

<p align="center">
    <br>
    <a href="https://github.com/safe-graph/GNN-FakeNews">
        <img src="https://github.com/safe-graph/GNN-FakeNews/blob/main/overview.png" width="1000"/>
    </a>
    <br>
<p>

## User Guide

All GNN-based fake news detection models are under the `\gnn_model` directory.
You can fine-tune each model according to arguments specified in the argparser of each model.

Since the UPFD framework is built upon the [PyG](https://github.com/rusty1s/pytorch_geometric), you can easily try other graph classification models
like [GIN](https://github.com/rusty1s/pytorch_geometric/blob/master/examples/mutag_gin.py) and [HGP-SL](https://github.com/cszhangzhen/HGP-SL)
under our dataset.

## Leader Board

To be updated.

### Politifact
| Model  | Feature  | Accuracy  | F1  |
|-------|--------|--------|--------|
| **[GNN-CL](https://arxiv.org/pdf/2007.03316.pdf)** |    |    |  |
| **[GCNFN](https://arxiv.org/pdf/1902.06673.pdf)** |   |  |   |
| **[BiGCN](https://arxiv.org/pdf/2001.06362.pdf)** |    |  | |
| **[UPFD-GCN](https://arxiv.org/pdf/1609.02907.pdf)** |  |  |  |
| **[UPFD-GAT](https://arxiv.org/pdf/1710.10903.pdf)** |  |  |  |
| **[UPFD-SAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)** |    |  |  |

### Gossipcop
| Model  | Feature  | Accuracy  | F1  |
|-------|--------|--------|--------|
| **[GNN-CL](https://arxiv.org/pdf/2007.03316.pdf)** |    |    |  |
| **[GCNFN](https://arxiv.org/pdf/1902.06673.pdf)** |   |  |   |
| **[BiGCN](https://arxiv.org/pdf/2001.06362.pdf)** |    |  | |
| **[UPFD-GCN](https://arxiv.org/pdf/1609.02907.pdf)** |  |  |  |
| **[UPFD-GAT](https://arxiv.org/pdf/1710.10903.pdf)** |  |  |  |
| **[UPFD-SAGE](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)** |    |  |  |

## How to Contribute
You are welcomed to submit your model, hyper-parameters, and results to this repo via create a pull request.
After verifying the results, your model will be added to the repo and the result will be updated to the leaderboard.
For other inquiries, please send email to [ytongdou@gmail.com](mailto:ytongdou@gmail.com).


