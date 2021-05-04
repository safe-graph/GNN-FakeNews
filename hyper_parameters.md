Model hyper-parameters are reported in the following table.

### Politifact
| model   | feature | epoch  | lr | emb_size | batch_num  |
|-------|--------|--------|--------|-----------|-----------|
| GNN-CL  | profile  | 60  | 0.001 | 128 | 128  |
| GCNFN  | content  | 100  | 0.001 | 128 | 128  |
| BiGCN  |  bert | 50  | 0.001 | 128 | 128  |
| UPFD-GAT  | bert  | 50  | 0.001 | 128 | 128  |
| UPFD-GCN  | spacy, profile  | 60  | 0.001 | 128 | 128  |
| UPFD-GCN  | bert  | 100  | 0.001 | 128 | 128  |
| UPFD-SAGE  | profile  | 70  | 0.01 | 128 | 128  |
| UPFD-SAGE  | spacy  | 45  | 0.01 | 128 | 128  |
| UPFD-SAGE  | bert  | 30  | 0.01 | 128 | 128  |
| UPFD-GCNFN  | profile  | 80  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | spacy  | 50  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | bert  | 60  | 0.001 | 128 | 128  |

### Gossipcop
| model   | feature | epoch  | lr | emb_size | batch_num  |
|-------|--------|--------|--------|-----------|-----------|
| GNN-CL  | profile  | 40  | 0.001 | 128 | 128  |
| GCNFN  | content  | 50  | 0.001 | 128 | 128  |
| BiGCN  |  bert | 35  | 0.001 | 128 | 128  |
| UPFD-GAT  | bert  | 30  | 0.001 | 128 | 128  |
| UPFD-GCN  |profile  | 50  | 0.01 | 128 | 128  |
| UPFD-GCN  | spacy, bert  | 50  | 0.001 | 128 | 128  |
| UPFD-SAGE  | profile  | 50  | 0.01 | 128 | 128  |
| UPFD-SAGE  | spacy bert  | 80  | 0.001 | 128 | 128  |
| UPFD-GCNFN  | profile, spacy, bert  | 50  | 0.001 | 128 | 128  |
