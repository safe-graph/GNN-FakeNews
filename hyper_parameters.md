The hyper-parameters are reported in the following table.

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
