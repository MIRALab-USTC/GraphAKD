#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import math
import time

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from models import GCN
import dgl.function as fn
import torch.nn as nn

device = None
in_feats, n_classes = None, None
epsilon = 1 - math.log(2)
# th.manual_seed(123)
import torch
import dgl


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
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 1.))

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
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 0.25))

    def forward(self, emb, summary):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        assert summary.shape[-1] == 1
        sim = sim @ summary
        return sim * self.scale


def wasserstein_dist_loss(logits_S, logits_T, temperature=1.0, p=1):
    prob_T = torch.softmax(logits_T / temperature, dim=-1)
    prob_S = torch.softmax(logits_S / temperature, dim=-1)
    cdf_T = prob_T.cumsum(-1)
    cdf_S = prob_S.cumsum(-1)
    wdist = torch.norm(torch.sort(cdf_T)[0] - torch.sort(cdf_S)[0], p=p, dim=-1) / cdf_T.shape[-1]
    return wdist.sum()


def gen_model(args):
    if args.use_labels:
        model = GCN(
            in_feats + n_classes, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear
        )
    else:
        model = GCN(in_feats, args.n_hidden, n_classes, args.n_layers, F.relu, args.dropout, args.use_linear)
    return model


def cross_entropy(x, labels):

    if labels.dim() != 1 and labels.shape[1] != 1:  # for Yelp
        return nn.BCEWithLogitsLoss()(x, labels)
    y = F.cross_entropy(x, labels[:, 0], reduction="none")
    y = th.log(epsilon + y) - math.log(epsilon)
    return th.mean(y)


def compute_acc(pred, labels, evaluator=None):
    if evaluator is None:
        y_pred = pred > 0
        y_true = labels > 0.5

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())

        try:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.
    return evaluator.eval({"y_pred": pred.argmax(dim=-1, keepdim=True), "y_true": labels})["acc"]


def add_labels(feat, labels, idx):
    if labels.shape[1] != 1: ## multi-class label like Yelp
        onehot = th.zeros([feat.shape[0], n_classes]).to(device)
        onehot[idx] = labels[idx]
    else: ## arxiv
        onehot = th.zeros([feat.shape[0], n_classes]).to(device)
        onehot[idx, labels[idx, 0]] = 1
    return th.cat([feat, onehot], dim=-1)


def adjust_learning_rate(optimizer, lr, epoch):
    if epoch <= 50:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * epoch / 50


def train(model, graph, labels, train_idx, optimizer, use_labels, *args):
    model.train()

    feat = graph.ndata["feat"]

    if use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate

        train_pred_idx = train_idx[mask]

    optimizer.zero_grad()
    pred = model(graph, feat)
    
    if len(args) != 0:
        label_loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        tea_logits, tea_emb, Discriminator_t, Discriminator_e, Discriminator_g, loss_dis, opt_D, nargs, epoch = args

        # ============================================
        #  Train Dis
        # ============================================
        if epoch % nargs.d_critic == 0:
            ## distinguish by Dl
            Discriminator_t.train()
            stu_logits = pred.detach()
            pos_z = Discriminator_t(tea_logits)
            neg_z = Discriminator_t(stu_logits)
            real_z = torch.sigmoid(pos_z[:, -1])
            fake_z = torch.sigmoid(neg_z[:, -1])
            ad_loss = loss_dis(real_z, torch.ones_like(real_z)) + loss_dis(fake_z, torch.zeros_like(fake_z))
            ds_loss = cross_entropy(pos_z[:, :-1][train_pred_idx], labels[train_pred_idx]) \
                      + cross_entropy(neg_z[:, :-1][train_pred_idx], labels[train_pred_idx])
            loss_D = 0.5 * (ad_loss + ds_loss)

            # distinguish by De
            pos_e = Discriminator_e(tea_emb, graph)
            neg_e = Discriminator_e(model.emb.detach(), graph)
            real_e = torch.sigmoid(pos_e)
            fake_e = torch.sigmoid(neg_e)
            ad_eloss = loss_dis(real_e, torch.ones_like(real_e)) + loss_dis(fake_e, torch.zeros_like(fake_e))
            #++++++++++++++++++++++++
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
            loss_D = ad_eloss + loss_D + ad_gloss1 + ad_gloss2
            #++++++++++++++++++++++++
            loss_G = loss_D

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
        
        # ============================================
        #  Train Stu
        # ============================================
        if epoch % nargs.g_critic == 0:
            ## distinguish by Dl
            Discriminator_t.eval()
            pos_z = Discriminator_t(tea_logits)
            neg_z = Discriminator_t(pred)
            fake_z = torch.sigmoid(neg_z[:, -1])
            ad_loss = loss_dis(fake_z, torch.ones_like(fake_z))
            ds_loss = cross_entropy(pos_z[:, :-1][train_pred_idx], labels[train_pred_idx]) \
                      + cross_entropy(neg_z[:, :-1][train_pred_idx], labels[train_pred_idx])
            l1_loss = torch.norm(pred - tea_logits, p=1) * 1. / len(tea_logits)
            loss_G = label_loss + 0.5 * (ds_loss + ad_loss) + l1_loss

            # distinguish by De
            neg_e = Discriminator_e(model.emb, graph)
            fake_e = torch.sigmoid(neg_e)
            ad_eloss = loss_dis(fake_e, torch.ones_like(fake_e))
            #++++++++++++++++++++++++
            tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
            neg_g = Discriminator_g(model.emb, tea_sum)
            fake_g = torch.sigmoid(neg_g)
            ad_gloss1 = loss_dis(fake_g, torch.ones_like(fake_g))

            stu_sum = torch.sigmoid(model.emb.mean(dim=0)).unsqueeze(-1)
            neg_g = Discriminator_g(tea_emb, stu_sum)
            pos_g = Discriminator_g(model.emb, stu_sum)
            real_g = torch.sigmoid(pos_g)
            fake_g = torch.sigmoid(neg_g)
            ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
            loss_G = loss_G + ad_eloss + ad_gloss1 + ad_gloss2
            #++++++++++++++++++++++++

            # optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

        return loss_G, pred
    else:
        loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        loss.backward()
        optimizer.step()
        return loss, pred


# for yelp and products
def cluster_train(model, cluster, feat, labels, mask, optimizer, args, *others):
    model.train()
    train_idx = torch.arange(cluster.num_nodes())[mask].to(feat.device)
    graph = cluster
    if args.use_labels:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate
        train_labels_idx = train_idx[mask]
        train_pred_idx = train_idx[~mask]

        feat = add_labels(feat, labels, train_labels_idx)
    else:
        mask_rate = 0.5
        mask = th.rand(train_idx.shape) < mask_rate
        train_pred_idx = train_idx[mask]
    
    optimizer.zero_grad()
    pred = model(cluster, feat)
    if len(others) != 0:
        label_loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        loss_G = label_loss
        tea_logits, tea_emb, Discriminator_t, Discriminator_e, Discriminator_g, loss_dis, opt_D, epoch = others

        # ============================================
        #  Train Dis
        # ============================================
        if epoch % args.d_critic == 0:
            loss_D = 0
            ## distinguish by Dl
            if not args.de_only:
                Discriminator_t.train()
                stu_logits = pred.detach()
                pos_z = Discriminator_t(tea_logits)
                neg_z = Discriminator_t(stu_logits)
                real_z = torch.sigmoid(pos_z[:, -1])
                fake_z = torch.sigmoid(neg_z[:, -1])
                ad_loss = loss_dis(real_z, torch.ones_like(real_z)) + loss_dis(fake_z, torch.zeros_like(fake_z))
                ds_loss = cross_entropy(pos_z[:, :-1][train_pred_idx], labels[train_pred_idx]) \
                        + cross_entropy(neg_z[:, :-1][train_pred_idx], labels[train_pred_idx])
                loss_D = 0.5 * (ad_loss + ds_loss)
            ## distinguish by De
            if not args.dl_only:
                Discriminator_e.train()
                pos_e = Discriminator_e(tea_emb, graph)
                neg_e = Discriminator_e(model.emb.detach(), graph)
                real_e = torch.sigmoid(pos_e)
                fake_e = torch.sigmoid(neg_e)
                ad_eloss = loss_dis(real_e, torch.ones_like(real_e)) + loss_dis(fake_e, torch.zeros_like(fake_e))
                #++++++++++++++++++++++++
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
                #++++++++++++++++++++++++

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

        # ============================================
        #  Train Stu
        # ============================================
        if epoch % args.g_critic == 0:
            loss_G = label_loss
            ## to fool Discriminator_t
            if not args.de_only:
                Discriminator_t.eval()
                neg_z = Discriminator_t(pred)
                fake_z = torch.sigmoid(neg_z[:, -1])
                ad_loss = loss_dis(fake_z, torch.ones_like(fake_z))
                ds_loss = cross_entropy(neg_z[:, :-1][train_pred_idx], labels[train_pred_idx])
                l1_loss = torch.norm(pred - tea_logits, p=1) * 1. / len(tea_logits)
                loss_G = loss_G + 0.5 * (ds_loss + ad_loss) + l1_loss
            
            ## to fool Discriminator_e
            if not args.dl_only:
                Discriminator_e.eval()
                neg_e = Discriminator_e(model.emb, graph)
                fake_e = torch.sigmoid(neg_e)
                ad_eloss = loss_dis(fake_e, torch.ones_like(fake_e))
                #++++++++++++++++++++++++
                tea_sum = torch.sigmoid(tea_emb.mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(model.emb, tea_sum)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss1 = loss_dis(fake_g, torch.ones_like(fake_g))

                stu_sum = torch.sigmoid(model.emb.mean(dim=0)).unsqueeze(-1)
                neg_g = Discriminator_g(tea_emb, stu_sum)
                pos_g = Discriminator_g(model.emb, stu_sum)
                real_g = torch.sigmoid(pos_g)
                fake_g = torch.sigmoid(neg_g)
                ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
                loss_G = loss_G + ad_eloss + ad_gloss1 + ad_gloss2
                #++++++++++++++++++++++++

            # optimizer.zero_grad()
            loss_G.backward()
            optimizer.step()

        return loss_G, pred
    else:
        loss = cross_entropy(pred[train_pred_idx], labels[train_pred_idx])
        loss.backward()
        optimizer.step()
        return loss, pred


@th.no_grad()
def evaluate(model, graph, labels, train_idx, val_idx, test_idx, use_labels, evaluator):
    model.eval()
    feat = graph.ndata["feat"]

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    pred = model(graph, feat)
    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
    )


@th.no_grad()
def cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, use_labels, evaluator, feat, data_name):
    model.eval()
    cuda = False if data_name == "ogbn-products" else True

    if use_labels:
        feat = add_labels(feat, labels, train_idx)

    perms = []
    preds = []
    for cluster in cluster_iterator:
        cluster = cluster.int().to(device)
        input_nodes = cluster.ndata['id']
        batch_feat = feat[input_nodes]
        pred = model(cluster, batch_feat) if cuda else model(cluster, batch_feat).cpu()
        perms.append(input_nodes)
        preds.append(pred)
    perm = th.cat(perms, dim=0)
    pred = th.cat(preds, dim=0)
    inv_perm=perm.sort()[1]
    if not cuda:
        inv_perm = inv_perm.cpu()
        pred = pred[inv_perm]
        labels = labels.cpu()
        train_idx, val_idx, test_idx = train_idx.cpu(), val_idx.cpu(), test_idx.cpu()
    else:
        pred = pred[inv_perm]
    

    train_loss = cross_entropy(pred[train_idx], labels[train_idx])
    val_loss = cross_entropy(pred[val_idx], labels[val_idx])
    test_loss = cross_entropy(pred[test_idx], labels[test_idx])

    return (
        compute_acc(pred[train_idx], labels[train_idx], evaluator),
        compute_acc(pred[val_idx], labels[val_idx], evaluator),
        compute_acc(pred[test_idx], labels[test_idx], evaluator),
        train_loss,
        val_loss,
        test_loss,
        pred
    )


def run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, n_running):
    # define model and optimizer
    model = gen_model(args)
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=100, verbose=True, min_lr=1e-3
    )

    # Get teacher knowledge
    import os
    kd_dir = '../../distilled'
    if args.dataset == 'ogbn-arxiv':
        kd_path = os.path.join(kd_dir, f'arxiv-knowledge.pth.tar')  # 73.14774808139415
    elif args.dataset == 'ogbn-products':
        kd_path = os.path.join(kd_dir, f'products-knowledge.pth.tar')
    elif args.dataset == 'yelp':
        kd_path = os.path.join(kd_dir, f'yelp-knowledge.pth.tar')
    assert os.path.isfile(kd_path), "Please download teacher knowledge first"
    knowledge = th.load(kd_path, map_location=device)
    tea_logits = knowledge['logits']
    tea_emb = knowledge['embedding']
    if 'perm' in knowledge.keys():
        perm = knowledge['perm']
        inv_perm = perm.sort()[1]
        tea_logits = tea_logits[inv_perm]
        tea_emb = tea_emb[inv_perm]
    print(f'Teacher Test ACC: {compute_acc(tea_logits[test_idx], labels[test_idx], evaluator)}')

    # Define Discriminators
    Discriminator_e = local_emb_D(n_hidden=args.n_hidden).to(device)
    Discriminator_g = global_emb_D(n_hidden=args.n_hidden).to(device)
    Discriminator_t = logits_D(n_classes, n_hidden=n_classes).to(device)
    opt_D = torch.optim.Adam([{"params": Discriminator_t.parameters()}, {"params": Discriminator_e.parameters()}, {"params": Discriminator_g.parameters()}],
                             lr=1e-2, weight_decay=5e-4)
    loss_dis = torch.nn.BCELoss()

    args.param_count = sum(param.numel() for param in model.parameters()) + sum(param.numel() for param in Discriminator_t.parameters()) + sum(param.numel() for param in Discriminator_e.parameters()) + sum(param.numel() for param in Discriminator_g.parameters())
    print(f"Param count: {args.param_count}")

    #+++++++++++++++++++++++++++++++++++++++++++++++
    # Create DataLoader for constructing blocks
    from functools import partial
    from torch.utils.data import DataLoader
    from sampler import ClusterIter, subgraph_collate_fn
    if args.dataset in ["ogbn-products", "yelp"]:
        nfeat = graph.ndata.pop('feat').to(device)
        cluster_iter_data = ClusterIter(args.dataset, graph, args.num_partitions) #'ogbn-products'
        mask = th.zeros(graph.num_nodes(), dtype=th.bool)
        mask[train_idx] = True
        graph.ndata['train_mask'] = mask
        graph.ndata['id'] = th.arange(graph.num_nodes())
        cluster_iterator = DataLoader(cluster_iter_data, batch_size=args.batch_size, shuffle=True,
                                    pin_memory=True, num_workers=4,
                                    collate_fn=partial(subgraph_collate_fn, graph))
    #+++++++++++++++++++++++++++++++++++++++++++++++
    graph = graph.int().to(device)

    # training loop
    total_time = 0
    best_val_acc, final_test_acc, best_val_loss = 0, 0, float("inf")

    accs, train_accs, val_accs, test_accs = [], [], [], []
    losses, train_losses, val_losses, test_losses = [], [], [], []


    for epoch in range(1, args.n_epochs + 1):
        tic = time.time()
        adjust_learning_rate(optimizer, args.lr, epoch)

        ## (๑>؂<๑) full batch GCN train arxiv
        if args.dataset in ["ogbn-arxiv"]:
            loss, pred = train(model, graph, labels, train_idx, optimizer, args.use_labels, tea_logits, tea_emb, Discriminator_t, Discriminator_e, Discriminator_g, loss_dis, opt_D, args, epoch)
            acc = compute_acc(pred[train_idx], labels[train_idx], evaluator)
            #+++++++++++++++++
            t_start = time.time()
            train_acc, val_acc, test_acc, train_loss, val_loss, test_loss = evaluate(
                model, graph, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator
            )
            if epoch == 1:
                t_end = time.time()
                test_time = t_end - t_start
                print(f'inference time: {test_time * 1e3} ms')
            #+++++++++++++++++
            if val_acc > best_val_acc:
                best_val_loss = val_loss
                best_val_acc = val_acc
                final_test_acc = test_acc
        ## mini batch cluster-GCN train products and yelp
        elif args.dataset in ["ogbn-products", "yelp"]:
            for step, cluster in enumerate(cluster_iterator):
                mask = cluster.ndata.pop('train_mask')
                if mask.sum() == 0:
                    continue
                cluster.edata.pop(dgl.EID)
                cluster = cluster.int().to(device)
                input_nodes = cluster.ndata['id']
                batch_feat = nfeat[input_nodes]
                batch_labels = labels[input_nodes]

                if args.role == 'vani':
                    loss, pred = cluster_train(model, cluster, batch_feat, batch_labels, mask, optimizer, args)
                elif args.role == 'stu':
                    loss, pred = cluster_train(model, cluster, batch_feat, batch_labels, mask, optimizer, args, tea_logits[input_nodes], tea_emb[input_nodes], Discriminator_t, Discriminator_e, Discriminator_g, loss_dis, opt_D, epoch)
                if step % args.log_every == 0:
                    acc = compute_acc(pred[mask], batch_labels[mask], evaluator)
                    print(
                    f"Epoch: {epoch}/{args.n_epochs} | Loss: {loss.item():.4f} | {step:3d}-th Cluster Train Acc: {acc:.4f}")
            #+++++++++++++++++ testing OOM
            if epoch == 1:
                t_start = time.time()
                train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator, nfeat, args.dataset)
                t_end = time.time()
                test_time = t_end - t_start
                print(f'inference time: {test_time * 1e3} ms')
                print(
                    f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f} s. \n"
                    f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                    f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
                )
            #+++++++++++++++++
            if epoch % args.log_every == 0:
                train_acc, val_acc, test_acc, train_loss, val_loss, test_loss, pred = cluster_eval(cluster_iterator, model, labels, train_idx, val_idx, test_idx, args.use_labels, evaluator, nfeat, args.dataset)
                if val_acc > best_val_acc:
                    best_val_loss = val_loss
                    best_val_acc = val_acc
                    final_test_acc = test_acc
        ##############################################

        lr_scheduler.step(loss)

        toc = time.time()
        total_time += toc - tic

        if epoch % args.log_every == 0:
            print(
                f"Run: {n_running}/{args.n_runs}, Epoch: {epoch}/{args.n_epochs}, Average epoch time: {total_time / epoch:.2f} s. \n"
                f"Loss: {loss.item():.4f}, Acc: {acc:.4f}\n"
                f"Train/Val/Test loss: {train_loss:.4f}/{val_loss:.4f}/{test_loss:.4f}\n"
                f"Train/Val/Test/Best val/Final test acc: {train_acc:.4f}/{val_acc:.4f}/{test_acc:.4f}/{best_val_acc:.4f}/{final_test_acc:.4f}"
            )

            for l, e in zip(
                [accs, train_accs, val_accs, test_accs, losses, train_losses, val_losses, test_losses],
                [acc, train_acc, val_acc, test_acc, loss, train_loss, val_loss, test_loss],
            ):
                l.append(e)

    
    print("*" * 50)
    print(args)
    print(f"Best val acc: {best_val_acc}, Final test acc: {final_test_acc}")
    print("*" * 50)

    return best_val_acc, final_test_acc


def count_parameters(args):
    model = gen_model(args)
    return sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


def main():
    global device, in_feats, n_classes

    argparser = argparse.ArgumentParser("GCN on OGBN-*", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--cpu", action="store_true", help="CPU mode. This option overrides --gpu.")
    argparser.add_argument("--gpu", type=int, default=0, help="GPU device ID.")
    argparser.add_argument("--n-runs", type=int, default=1, help="running times")
    argparser.add_argument("--n-epochs", type=int, default=1000, help="number of epochs")
    argparser.add_argument("--use-labels", action="store_true", help="Use labels in the training set as input features.")
    argparser.add_argument("--use-linear", action="store_true", help="Use linear layer.")
    argparser.add_argument("--lr", type=float, default=0.005, help="learning rate")
    argparser.add_argument("--n-layers", type=int, default=3, help="number of layers")
    argparser.add_argument("--n-hidden", type=int, default=256, help="number of hidden units")
    argparser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    argparser.add_argument("--wd", type=float, default=0, help="weight decay")
    argparser.add_argument("--log-every", type=int, default=20, help="log every LOG_EVERY epochs")
    argparser.add_argument("--seed", type=int, default=2022, help="random seed")
    argparser.add_argument("--alpha", type=float, default=0., help="weight of feat loss")
    argparser.add_argument("--beta", type=float, default=0., help="reweight of feat loss")
    argparser.add_argument("--atemp", type=float, default=1.0, help="temperature of feat loss")
    argparser.add_argument("-d", "--dataset", type=str, default='ogbn-arxiv', help="Dataset name ('ogbn-products', 'ogbn-arxiv', 'yelp').")
    # extra added
    argparser.add_argument("--num_partitions", type=int, default=200, help="num of subgraphs")
    argparser.add_argument("--batch-size", type=int, default=32, help="batch size")
    argparser.add_argument("--d-critic", type=int, default=1, help="train discriminator")
    argparser.add_argument("--g-critic", type=int, default=1, help="train generator")
    argparser.add_argument("--role", type=str, default="vani", choices=['stu', 'vani'])

    args = argparser.parse_args()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print(args)

    if args.cpu:
        device = th.device("cpu")
    else:
        device = th.device("cuda:%d" % args.gpu)

    # load data
    if args.dataset == 'yelp':
        from torch_geometric.datasets import Yelp
        import torch_geometric.transforms as T
        root = '../../datasets'
        pyg_data = Yelp(f'{root}/YELP', pre_transform=T.ToSparseTensor())[0]  # replace edge_index with adj
        labels = pyg_data.y
        adj = pyg_data.adj_t.to_torch_sparse_coo_tensor()
        u, v = adj.coalesce().indices()
        g = dgl.graph((u, v))
        g.ndata['feat'] = pyg_data.x
        idx = torch.arange(g.num_nodes())
        train_idx, val_idx, test_idx = idx[pyg_data.train_mask], idx[pyg_data.val_mask], idx[pyg_data.test_mask]
        n_classes = labels.size(1)  # multi-label classification
        graph = g
        evaluator = None
    else:  # arxiv or products
        root = '../../datasets'
        data = DglNodePropPredDataset(name=args.dataset, root=f'{root}/OGB/')
        evaluator = Evaluator(name=args.dataset)
        splitted_idx = data.get_idx_split()
        train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
        graph, labels = data[0]

    # add reverse edges
    srcs, dsts = graph.all_edges()
    graph.add_edges(dsts, srcs)

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    if args.dataset != 'yelp':
        graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    if args.dataset == 'yelp':
        n_classes = labels.size(1)  # multi-label classification

    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    test_idx = test_idx.to(device)
    labels = labels.to(device)

    # run
    val_accs = []
    test_accs = []

    for i in range(args.n_runs):
        th.manual_seed(args.seed)
        val_acc, test_acc = run(args, graph, labels, train_idx, val_idx, test_idx, evaluator, i)
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        args.seed += 1

    print(f"Runned {args.n_runs} times")
    print("Val Accs:", val_accs)
    print("Test Accs:", test_accs)
    print(f"Average val accuracy: {np.mean(val_accs)} ± {np.std(val_accs)}")
    print(f"Average test accuracy on {args.dataset}: {np.mean(test_accs)} ± {np.std(test_accs)}")
    print(f"Number of params: {args.param_count}")


if __name__ == "__main__":
    main()
