# Student GNNs for ogbg-mol*
Before downloading datasets, please add the following code at Line 135 of `ogb.graphproppred.PygGraphPropPredDataset`
```python
#++++++++++++++++++++++++ add mol index
for i, g in enumerate(data_list):
    g.id = torch.tensor([i])
#++++++++++++++++++++++++
```

## ogbg-molhiv

```bash
cd graph-level/stu-gnn
export CUDA_VISIBLE_DEVICES=0 

python main.py --dataset ogbg-molhiv --gnn gcn --filename gcnout --emb_dim=256 --role=stu --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001

python main.py --dataset ogbg-molhiv --gnn gin --filename ginout --emb_dim=256 --role=stu --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001
```

## ogbg-molpcba

```bash
cd graph-level/stu-gnn
export CUDA_VISIBLE_DEVICES=0 

python main.py --dataset ogbg-molpcba --gnn gcn --filename gcnout --emb_dim=1024 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001 --role=stu

python main.py --dataset ogbg-molpcba --gnn gin --filename ginout --emb_dim=1024 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.005 --wd=0 --role=stu
```

