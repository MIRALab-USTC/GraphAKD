# Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation

This is the code of paper 
**Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation**. 
Huarui He, Jie Wang, Zhanqiu Zhang, Feng Wu. SIGKDD 2022. 
[[arXiv](https://arxiv.org/abs/2205.11678)]

## Requirements
- python 3.7.3
- torch 1.9.1
- dgl 0.9.1
- ogb 1.3.4
- torch-geometric 2.1.0
- gdown 4.5.1
<!-- dgl-cu113                 0.9.1  -->
<!-- torch                     1.9.1+cu111 -->
<!-- https://github.com/snap-stanford/ogb/issues/346   A quick workaround is to uninstall outdated after ogb is installed. -->

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
@inproceedings{KDD22_GraphAKD,
  author={Huarui He and Jie Wang and Zhanqiu Zhang and Feng Wu},
  booktitle={Proc. of SIGKDD},
  title={Compressing Deep Graph Neural Networks via Adversarial Knowledge Distillation},
  year={2022}
}
```

## Acknowledgement
We refer to the code of [DGL](https://github.com/dmlc/dgl). Thanks for their contributions.
