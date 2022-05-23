### Student GCN for cora, citeseer, pubmed; flickr, and reddit

Requires DGL no later than 0.6.x versions as new-version DGL will use `dgl.reorder_graph` to reorder the three citation networks.

```bash
cd node-level/stu-gcn
export CUDA_VISIBLE_DEVICES=0

python3 train.py --dataset cora --n-hidden 64 --n-epochs 600 --role=stu --lr=0.05 --d-critic=10 --n-runs=10

python3 train.py --dataset citeseer --n-epochs 1000 --role=stu --d-critic=8 --n-runs=10

python3 train.py --dataset pubmed --n-epochs 600 --role=stu --d-critic=10 --n-runs=10

python3 train.py --dataset flickr --n-epochs 2000 --role=stu --d-critic=6 --lr=0.005 --n-runs=10

python3 train.py --dataset reddit --n-epochs 1500 --role=stu --lr=0.01 --dropout=0.1 --n-runs=10

```

### Student GCN for ogbn-arxiv

```bash
cd node-level/stu-cluster-gcn
export CUDA_VISIBLE_DEVICES=0

python3 train.py --use-linear -d=ogbn-arxiv --n-epochs=2000 --role=stu --use-labels --d-critic=30 --lr=0.01 --dropout=0.1  --n-runs=10
```

### Student Cluster-GCN for yelp, and ogbn-products

```bash
cd node-level/stu-cluster-gcn
export CUDA_VISIBLE_DEVICES=0 

python3 train.py --use-linear -d=yelp --n-epochs=200 --n-hidden=512 --role=stu --lr=0.005 --num_partition=120 --batch-size=32 --dropout=0.2

python3 train.py --use-linear -d=ogbn-products --n-epochs=400 --n-hidden=512 --role=stu --num_partition=160 --batch-size=4 --dropout=0.1
```