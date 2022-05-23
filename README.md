# Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation

This is the code of paper 
**Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation**. 
Huarui He, Jie Wang, Zhanqiu Zhang, Feng Wu. SIGKDD 2022. 
<!-- [[arXiv](https://arxiv.org/abs/2202.05679)] -->

## Requirements
- python 3.7.3
- torch 1.9.1
- dgl 0.6
- ogb 1.3.2
- torch-geometric 2.0.2
- gdown 4.4.0



## Reproduce the Results
First, download teacher knowledge from Google Drive
```bash
python download_teacher_knowledge.py --data_name=<dataset>
python download_teacher_knowledge.py --data_name=cora
```
Second, pleaes run the commands in `node-level/README.md` or `graph-level/README.md` to reproduce the results.

## File tree
```
GraphAKD
├─ README.md
├─ download_teacher_knowledge.py
├─ datasets
│  └─ ...
├─ distilled
│  ├─ cora-knowledge.pth.tar
│  └─ ...
├─ graph-level
│  ├─ README.md
│  └─ stu-gnn
│     ├─ conv.py
│     ├─ gnn.py
│     └─ main.py
└─ node-level
   ├─ README.md
   ├─ stu-cluster-gcn
   │  ├─ dataset
   │  │  ├─ ogbn-products_160.npy
   │  │  └─ yelp_120.npy
   │  ├─ gcnconv.py
   │  ├─ models.py
   │  ├─ sampler.py
   │  └─ train.py
   └─ stu-gcn
      ├─ gcn.py
      ├─ gcnconv.py
      └─ train.py
```


## Citation
If you find this code useful, please consider citing the following paper.
```
TBA
```

<!-- ```
@inproceedings{WWW22_GCN4KGC,
 author = {Zhanqiu Zhang and Jie Wang and Jieping Ye and Feng Wu},
 booktitle = {The Web Conference 2022},
 title = {Rethinking Graph Convolutional Networks in Knowledge Graph Completion},
 year = {2022}
}
``` -->

## Acknowledgement
We refer to the code of [DGL](https://github.com/dmlc/dgl). Thanks for their contributions.
