# UPFD
Code for SIGIR2021 submission User Preference-aware Fake News Detection

/Detector includes the implementation of our models and all baselines. The congfiguration can be setup with model arguments for each model. Note that the "hand" feature in model arguments represents the "Profile" feature.



To run the code, you need to install Python>=3.6 and following packages:

```
PyTorch>=1.6
PyTorch_Geometric>=1.6.1
keras>=2.2.4
scikit-learn>=0.22.1
tqdm>=4.31.1
numpy>=1.19.4
scipy>=1.5.2
```


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
