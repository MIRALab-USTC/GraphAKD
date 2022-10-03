import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

from gcn import GCN
import random
import os
import dgl.function as fn
import warnings
warnings.filterwarnings('ignore')


def set_random_seed(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    dgl.random.seed(seed)


def compute_micro_f1(logits, y, mask=None):
    if mask is not None:
        logits, y = logits[mask], y[mask]

    if y.dim() == 1:
        return int(logits.argmax(dim=-1).eq(y).sum()) / y.size(0)
    else:
        y_pred = logits > 0
        y_true = y > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.


def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


class logits_D(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.n_hidden, self.n_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.n_hidden, self.n_class+1, bias=False)

    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        out = self.relu(out)
        dist = self.lin2(out)
        return dist


class local_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(local_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb, g):
        emb = F.normalize(emb, p=2)
        g.ndata['e'] = emb
        g.ndata['ew'] = emb @ torch.diag(self.d)
        g.apply_edges(fn.u_dot_v('ew', 'e', 'z'))
        pair_dis = g.edata['z']
        return pair_dis * self.scale

class global_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.5))

    def forward(self, emb, summary):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        assert summary.shape[-1] == 1
        sim = sim @ summary
        return sim * self.scale


def run(args, g, n_classes, cuda, n_running):
    set_random_seed(args)
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_edges = g.number_of_edges()

    # load teacher knowledge
    if args.role == 'stu':
        kd_path = os.path.join(args.kd_dir, args.dataset + f'-knowledge.pth.tar')
        assert os.path.isfile(kd_path), "Please download teacher knowledge first"
        knowledge = torch.load(kd_path, map_location=g.device)
        tea_logits = knowledge['logits']
        tea_emb = knowledge['embedding']
        if 'perm' in knowledge.keys() and args.dataset in ['arxiv', 'reddit']:
            perm = knowledge['perm']
            inv_perm = perm.sort()[1]
            tea_logits = tea_logits[inv_perm]
            tea_emb = tea_emb[inv_perm]

        test_acc = compute_micro_f1(tea_logits, labels, test_mask)  # for val
        print(f'Teacher Test SCORE: {test_acc:.3%}')

    # create GCN model as Generator
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)

    if labels.dim() == 1:
        loss_fcn = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError('Unknown dataset with wrong labels: {}'.format(args.dataset))

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    Discriminator_e = local_emb_D(n_hidden=args.n_hidden)
    Discriminator_g = global_emb_D(n_hidden=args.n_hidden)
    Discriminator = logits_D(n_classes, n_hidden=n_classes)
    if cuda:
        model.cuda()
        Discriminator.cuda()
        Discriminator_e.cuda()
        Discriminator_g.cuda()
    opt_D = torch.optim.Adam([{"params": Discriminator.parameters()}, {"params": Discriminator_e.parameters()}, {"params": Discriminator_g.parameters()}],
                             lr=args.lr, weight_decay=args.weight_decay)
    loss_dis = torch.nn.BCELoss()

    param_count = sum(param.numel() for param in model.parameters()) + sum(param.numel() for param in Discriminator.parameters()) + sum(param.numel() for param in Discriminator_e.parameters()) + sum(param.numel() for param in Discriminator_g.parameters())

    dur = []
    log_every = 30
    best_eval_acc, final_test_acc, best_val_loss = 0, 0, float("inf")
    for epoch in range(args.n_epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        logits = model(features)
        label_loss = loss_fcn(logits[train_mask], labels[train_mask])
        if args.role == 'stu':
            # ============================================
            #  Train Dis
            # ============================================
            if epoch % args.d_critic == 0:
                loss_D = 0
                ## distinguish by Dl
                Discriminator.train()
                stu_logits = logits.detach()
                pos_z = Discriminator(tea_logits)
                neg_z = Discriminator(stu_logits)
                real_z = torch.sigmoid(pos_z[:, -1])
                fake_z = torch.sigmoid(neg_z[:, -1])
                ad_loss = loss_dis(real_z, torch.ones_like(real_z)) + loss_dis(fake_z, torch.zeros_like(fake_z))
                ds_loss = loss_fcn(pos_z[:, :-1][train_mask], labels[train_mask]) + loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])
                loss_D = 0.5 * (ad_loss + ds_loss)

                # distinguish by De
                Discriminator_e.train()
                pos_e = Discriminator_e(tea_emb, g)
                neg_e = Discriminator_e(model.emb.detach(), g)
                real_e = torch.sigmoid(pos_e)
                fake_e = torch.sigmoid(neg_e)
                ad_eloss = loss_dis(real_e, torch.ones_like(real_e)) + loss_dis(fake_e, torch.zeros_like(fake_e))
                # distinguish by Dg
                Discriminator_g.train()
                tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
                pos_g = Discriminator_g(tea_emb, tea_sum)
                neg_g = Discriminator_g(model.emb.detach(), tea_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))

                stu_sum = torch.sigmoid(model.emb.detach().mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(tea_emb, stu_sum)
                pos_g = Discriminator_g(model.emb.detach(), stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
                loss_D = loss_D + ad_eloss + ad_gloss1 + ad_gloss2

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()

            # ============================================
            #  Train Stu
            # ============================================
            if epoch % args.g_critic == 0:
                loss_G = label_loss
                ## to fool Discriminator_l
                Discriminator.eval()
                pos_z = Discriminator(tea_logits)
                neg_z = Discriminator(logits)
                fake_z = torch.sigmoid(neg_z[:, -1])
                ad_loss = loss_dis(fake_z, torch.ones_like(fake_z))
                ds_loss = loss_fcn(neg_z[:, :-1][train_mask], labels[train_mask])  # right one
                l1_loss = torch.norm(logits - tea_logits, p=1) * 1. / len(tea_logits)
                loss_G = loss_G + 0.5 * (ds_loss + ad_loss) + l1_loss

                ## to fool Discriminator_e
                Discriminator_e.eval()
                neg_e = Discriminator_e(model.emb, g)
                fake_e = torch.sigmoid(neg_e)
                ad_eloss = loss_dis(fake_e, torch.ones_like(fake_e))
                ## to fool Discriminator_g
                Discriminator_g.eval()
                tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(model.emb, tea_sum)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = loss_dis(fake_g, torch.ones_like(fake_g))

                stu_sum = torch.sigmoid(model.emb.mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(tea_emb, stu_sum)
                pos_g = Discriminator_g(model.emb, stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = loss_dis(real_g, torch.zeros_like(real_g)) + loss_dis(fake_g, torch.ones_like(fake_g))
                loss_G = loss_G + ad_eloss + ad_gloss1 + ad_gloss2

                optimizer.zero_grad()
                loss_G.backward()
                optimizer.step()

        else:
            loss = loss_fcn(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_D = loss

        if epoch >= 3:
            dur.append(time.time() - t0)

        val_loss = loss_fcn(logits[val_mask], labels[val_mask])
        eval_acc = compute_micro_f1(logits, labels, val_mask) 
        test_acc = compute_micro_f1(logits, labels, test_mask)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_eval_acc = eval_acc
            final_test_acc = test_acc
        if epoch % log_every == 0:
            print(f"Run: {n_running}/{args.n_runs} | Epoch {epoch:05d} | Time(s) {np.mean(dur):.4f} | Loss {loss_D.item():.4f} | "
            f"Val {eval_acc:.4f} | Test {test_acc:.4f} | Best Test {final_test_acc:.4f} | ETputs(KTEPS) {n_edges / np.mean(dur) / 1000:.2f}")

    print()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(args)
    print(f"Param count: {param_count}")
    print(f"Test accuracy on {args.dataset}: {final_test_acc:.2%}\n")
    return best_eval_acc, final_test_acc


def main(args):
    # load and preprocess dataset
    if args.dataset == 'cora':
        data = CoraGraphDataset()
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset()
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset()
    elif args.dataset == 'flickr':
        from torch_geometric.datasets import Flickr
        import torch_geometric.transforms as T
        pyg_data = Flickr(root=f'{args.data_dir}/Flickr', pre_transform=T.ToSparseTensor())[0]  # replace edge_index with adj
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))

        g.ndata['feat'] = pyg_data.x
        g.ndata['label'] = pyg_data.y
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        n_classes = pyg_data.y.max().item() + 1
    elif args.dataset == 'reddit':
        from torch_geometric.datasets import Reddit2
        import torch_geometric.transforms as T
        pyg_data = Reddit2(f'{args.data_dir}/Reddit2', pre_transform=T.ToSparseTensor())[0]
        pyg_data.x = (pyg_data.x - pyg_data.x.mean(dim=0)) / pyg_data.x.std(dim=0)
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))
        g.ndata['feat'] = pyg_data.x
        g.ndata['label'] = pyg_data.y
        g.ndata['train_mask'] = pyg_data.train_mask
        g.ndata['val_mask'] = pyg_data.val_mask
        g.ndata['test_mask'] = pyg_data.test_mask
        n_classes = pyg_data.y.max().item() + 1
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.dataset not in ['reddit', 'yelp', 'flickr', 'corafull']:
        g = data[0]
        n_classes = data.num_labels
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        g = g.int().to(args.gpu)
    
    # add self loop
    if args.self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)

    # normalization
    degs = g.in_degrees().clamp(min=1).float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    if cuda:
        norm = norm.cuda()
    g.ndata['norm'] = norm.unsqueeze(1)

    # run
    val_accs = []
    test_accs = []
    for i in range(args.n_runs):
        val_acc, test_acc = run(args, g, n_classes, cuda, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        args.seed += 1

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy on {args.dataset}: {np.mean(test_accs)} ± {np.std(test_accs)}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=600, help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256, help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    parser.add_argument("--role", type=str, default="vani", choices=['stu', 'vani'])
    parser.add_argument("--data_dir", type=str, default='../../datasets')
    parser.add_argument("--kd_dir", type=str, default='../../distilled')
    parser.set_defaults(self_loop=True)

    parser.add_argument("--d-critic", type=int, default=1, help="train discriminator")
    parser.add_argument("--g-critic", type=int, default=1, help="train generator")
    parser.add_argument("--n-runs", type=int, default=1, help="running times")
    args = parser.parse_args()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(args)

    main(args)


"""
CiteSeer

Runned 10 times
Val Accs: [0.726, 0.72, 0.714, 0.702, 0.742, 0.724, 0.72, 0.714, 0.734, 0.712]
Test Accs: [0.727, 0.718, 0.723, 0.723, 0.73, 0.744, 0.737, 0.731, 0.733, 0.729]
Average val accuracy: 0.7208 ± 0.01088852607105297
Average test accuracy on citeseer: 0.7295 ± 0.007102816342831912
"""