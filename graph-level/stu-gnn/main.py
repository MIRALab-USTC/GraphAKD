import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import json
from tqdm import tqdm
import argparse
import numpy as np
from gnn import GNN
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator


class logits_D(nn.Module):
    def __init__(self, n_class, n_hidden):
        super(logits_D, self).__init__()
        self.n_class = n_class
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.n_class, self.n_hidden) # assert n_class==n_hidden
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

    def forward(self, emb, batch):
        emb = F.normalize(emb, p=2)
        (u, v) = batch.edge_index
        euw = emb[u] @ torch.diag(self.d)
        pair_dis = euw @ emb[v].t()
        return torch.diag(pair_dis) * self.scale


class global_emb_D(nn.Module):
    def __init__(self, n_hidden):
        super(global_emb_D, self).__init__()
        self.n_hidden = n_hidden
        self.d = nn.Parameter(torch.ones(size=(n_hidden, )))
        self.scale = nn.Parameter(torch.full(size=(1, ), fill_value= 1.))

    def forward(self, emb, summary, batch):
        emb = F.normalize(emb, p=2)
        sim = emb @ torch.diag(self.d)
        sims = []
        for i, s in enumerate(summary):
            pre, post = batch.ptr[i], batch.ptr[i+1]
            sims.append(sim[pre:post] @ s.unsqueeze(-1))
        sim = torch.cat(sims, dim=0).squeeze(-1)
        return sim * self.scale


def load_knowledge(kd_path, device): # load teacher knowledge
    assert os.path.isfile(kd_path), "Please download teacher knowledge first"
    knowledge = torch.load(kd_path, map_location=device)
    tea_logits = knowledge['logits'].float()
    tea_h = knowledge['h-embedding']
    tea_g = knowledge['g-embedding']
    new_ptr = knowledge['ptr']
    return tea_logits, tea_h, tea_g, new_ptr

cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()

def train(model, device, loader, optimizer, task_type, others):
    model.train()
    (tea_logits, tea_h, tea_g, new_ptr, Discriminator_e, Discriminator_g, Discriminator_l, opt_D, loss_dis, epoch, args, train_ids) = others

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            # pred = model(batch)  # torch.Size([32, 1])
            pred, stu_bh, stu_bg = model(batch)  # torch.Size([32, 1]), [#nodes, out_dim], [#graphs, out_dim]
            optimizer.zero_grad()
            ## ignore nan targets (unlabeled) when computing training loss.
            is_labeled = batch.y == batch.y
            if "classification" in task_type:  # 'binary classification'
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            #++++++++++++++++++++++++
            if args.role == 'stu':
                new_pre = new_ptr[:-1]
                new_post = new_ptr[1:]
                new_ids = [(train_ids==vid).nonzero().item() for vid in batch.id]
                bids = torch.tensor(new_ids, device=new_ptr.device)
                bpre = new_pre[bids]
                bpost = new_post[bids]
                bnid = torch.cat([torch.arange(pre, post) for pre, post in list(zip(*[bpre, bpost]))], dim=0)
                tea_bh = tea_h[bnid].to(device)
                tea_bg = tea_g[bids].to(device)
                tea_by = tea_logits[bids].to(device)

                # ============================================
                #  Train Dis
                # ============================================
                if epoch % args.d_critic == 0:
                    loss_D = 0
                    ## distinguish by Dl
                    Discriminator_l.train()
                    stu_logits = pred.detach()
                    pos_z = Discriminator_l(tea_by)
                    neg_z = Discriminator_l(stu_logits)
                    real_z = torch.sigmoid(pos_z[:, -1])
                    fake_z = torch.sigmoid(neg_z[:, -1])
                    ad_loss = loss_dis(real_z, torch.ones_like(real_z)) + loss_dis(fake_z, torch.zeros_like(fake_z))
                    ds_loss = cls_criterion(pos_z[:, :-1][is_labeled], batch.y.to(torch.float32)[is_labeled]) + cls_criterion(neg_z[:, :-1][is_labeled], batch.y.to(torch.float32)[is_labeled])
                    loss_D = 0.5 * (ad_loss + ds_loss)

                    # distinguish by De
                    Discriminator_e.train()
                    pos_e = Discriminator_e(tea_bh, batch)
                    neg_e = Discriminator_e(stu_bh.detach(), batch)
                    real_e = torch.sigmoid(pos_e)
                    fake_e = torch.sigmoid(neg_e)
                    ad_eloss = loss_dis(real_e, torch.ones_like(real_e)) + loss_dis(fake_e, torch.zeros_like(fake_e))
                    #++++++++++++++++++++++++
                    # distinguish by Dg
                    Discriminator_g.train()
                    Discriminator_g.train()
                    tea_bg = torch.sigmoid(tea_bg)
                    pos_g = Discriminator_g(tea_bh, tea_bg, batch)
                    neg_g = Discriminator_g(stu_bh.detach(), tea_bg, batch)
                    real_g = torch.sigmoid(pos_g)
                    fake_g = torch.sigmoid(neg_g)
                    ad_gloss1 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))

                    stu_bg = torch.sigmoid(stu_bg)
                    neg_g = Discriminator_g(tea_bh, stu_bg.detach(), batch)
                    pos_g = Discriminator_g(stu_bh.detach(), stu_bg.detach(), batch)
                    real_g = torch.sigmoid(pos_g)
                    fake_g = torch.sigmoid(neg_g)
                    ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
                    loss_D = ad_eloss + loss_D + ad_gloss1 + ad_gloss2
                    #++++++++++++++++++++++++

                    opt_D.zero_grad()
                    loss_D.backward()
                    opt_D.step()
                # ============================================
                #  Train Stu
                # ============================================
                if epoch % args.g_critic == 0:
                    loss_G = loss
                    ## to fool Discriminator_l
                    Discriminator_l.eval()
                    pos_z = Discriminator_l(tea_by)
                    neg_z = Discriminator_l(pred)
                    fake_z = torch.sigmoid(neg_z[:, -1])
                    ad_loss = loss_dis(fake_z, torch.ones_like(fake_z))
                    ds_loss = cls_criterion(neg_z[:, :-1][is_labeled], batch.y.to(torch.float32)[is_labeled])
                    l1_loss = torch.norm(pred - tea_by, p=1) * 1. / len(batch.id)
                    loss_G = loss_G + 0.5 * (ds_loss + ad_loss) + l1_loss

                    ## to fool Discriminator_e
                    Discriminator_e.eval()
                    neg_e = Discriminator_e(stu_bh, batch)
                    fake_e = torch.sigmoid(neg_e)
                    ad_eloss = loss_dis(fake_e, torch.ones_like(fake_e))
                    #++++++++++++++++++++++++Start
                    ## to fool Discriminator_g
                    Discriminator_g.eval()
                    tea_bg = torch.sigmoid(tea_bg)
                    neg_g = Discriminator_g(stu_bh, tea_bg, batch)
                    fake_g = torch.sigmoid(neg_g)
                    ad_gloss1 = loss_dis(fake_g, torch.ones_like(fake_g))

                    stu_bg = torch.sigmoid(stu_bg)
                    neg_g = Discriminator_g(tea_bh, stu_bg, batch)
                    pos_g = Discriminator_g(stu_bh, stu_bg, batch)
                    real_g = torch.sigmoid(pos_g)
                    fake_g = torch.sigmoid(neg_g)
                    # ad_gloss2 = loss_dis(real_g, torch.ones_like(real_g)) + loss_dis(fake_g, torch.zeros_like(fake_g))
                    ad_gloss2 = loss_dis(real_g, torch.zeros_like(real_g)) + loss_dis(fake_g, torch.ones_like(fake_g))
                    #++++++++++++++++++++++++
                    loss_G = loss_G + ad_eloss + ad_gloss1 + ad_gloss2

                    optimizer.zero_grad()
                    loss_G.backward()
                    optimizer.step()
            else:
                loss.backward()
                optimizer.step()

def eval(model, device, loader, evaluator, distill=False):
    model.eval()
    y_true = []
    y_pred = []
    #++++++++++++++++++++++++
    y_ids = []
    ptrs = []
    #++++++++++++++++++++++++

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())
            #++++++++++++++++++++++++Start
            if distill:
                y_ids.append(batch.id.cpu())
                if len(ptrs) == 0:
                    ptrs.append(batch.ptr.cpu())
                else:
                    ptr = batch.ptr.cpu() + ptrs[-1][-1]
                    ptrs.append(ptr)
            #++++++++++++++++++++++++End

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    #++++++++++++++++++++++++Start
    if distill:
        y_id = torch.cat(y_ids, dim=0)
        ptr = torch.cat(ptrs, dim=0).unique()
        inv_perm = y_id.sort()[1]
        ptr_diff = torch.tensor(np.diff(ptr.numpy()))
        inv_ptr = ptr_diff[inv_perm].cumsum(dim=0)
        inv_ptr = torch.cat([torch.tensor([0]), inv_ptr], dim=0)
    #++++++++++++++++++++++++

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    if distill:
        return evaluator.eval(input_dict), inv_ptr

    return evaluator.eval(input_dict)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual', help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5, help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5, help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=48, help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv", help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full", help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="output", help='filename to output result (default: )')
    parser.add_argument("--role", type=str, default="vani", choices=['stu', 'vani'])
    parser.add_argument("--data_dir", type=str, default='../../datasets')
    parser.add_argument("--kd_dir", type=str, default='../../distilled')
    parser.add_argument("--d-critic", type=int, default=1, help="critic iteration")
    parser.add_argument("--g-critic", type=int, default=1, help="critic iteration")
    parser.add_argument("--lr", type=float, default=5e-3, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--seed", type=int, default=2022, help="random seed")
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygGraphPropPredDataset(name = args.dataset, root=f'{args.data_dir}/OGB/')

    if args.feature == 'full':
        pass 
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    #++++++++++++++++++++++++Load knowledge
    if args.role != 'vani':
        data_name=args.dataset.split('-')[1].upper()
        kd_path = os.path.join(args.kd_dir, data_name + f'-knowledge.pth.tar')
        if args.dataset == 'ogbg-molpcba':
            tea_logits, tea_h, tea_g, new_ptr = load_knowledge(kd_path, device='cpu')
        elif args.dataset == 'ogbg-molhiv':
            tea_logits, tea_h, tea_g, new_ptr = load_knowledge(kd_path, device=device)
        y_true = dataset.data.y[split_idx["train"]]
        input_dict = {"y_true": y_true, "y_pred": tea_logits}
        print(f'Teacher performance on Training set: {evaluator.eval(input_dict)}')
    else:
        tea_logits, tea_h, tea_g, new_ptr = None, None, None, None

    Discriminator_e = local_emb_D(n_hidden=args.emb_dim).to(device)
    Discriminator_g = global_emb_D(n_hidden=args.emb_dim).to(device)
    Discriminator_l = logits_D(n_class=dataset.num_tasks, n_hidden=dataset.num_tasks).to(device)
    opt_D = torch.optim.Adam([{"params": Discriminator_l.parameters()}, {"params": Discriminator_e.parameters()}, {"params": Discriminator_g.parameters()}], lr=1e-2, weight_decay=5e-4)
    loss_dis = torch.nn.BCELoss()
    #++++++++++++++++++++++++

    valid_curve = []
    test_curve = []
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        others = (tea_logits, tea_h, tea_g, new_ptr, Discriminator_e, Discriminator_g, Discriminator_l, opt_D, loss_dis, epoch, args, split_idx["train"].to(device))
        train(model, device, train_loader, optimizer, dataset.task_type, others)

        print('Evaluating...')
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Validation': valid_perf, 'Test': test_perf})

        valid_curve.append(valid_perf[dataset.eval_metric])
        test_curve.append(test_perf[dataset.eval_metric])

    if 'classification' in dataset.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))

    print('Finished training!')
    print(args)
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        record = dict(zip(valid_curve, test_curve))
        json_str = json.dumps(record, indent=4)
        with open(args.filename + '.json', 'w') as json_file:
            json_file.write(json_str)
        print(f'JSON Done!')


if __name__ == "__main__":
    main()


## ogbg-molhiv
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbg-molhiv --gnn gcn --filename gcnout --emb_dim=256 --role=stu --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbg-molhiv --gnn gin --filename ginout --emb_dim=256 --role=stu --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001

## ogbg-molpcba
# CUDA_VISIBLE_DEVICES=0 python main.py --dataset ogbg-molpcba --gnn gcn --filename gcnout --emb_dim=1024 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001 --role=stu
# CUDA_VISIBLE_DEVICES=1 python main.py --dataset ogbg-molpcba --gnn gin --filename ginout --emb_dim=1024 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.005 --wd=0 --role=stu

# CUDA_VISIBLE_DEVICES=4 python main.py --dataset ogbg-molpcba --gnn gcn --filename gcnout --emb_dim=256 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001
# CUDA_VISIBLE_DEVICES=4 python main.py --dataset ogbg-molpcba --gnn gin --filename ginout --emb_dim=256 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.005 --wd=0
#(graph) hrh@amax:~/Graph/GraphAKD_code/graph-level/mol-stu$ CUDA_VISIBLE_DEVICES=4 python main.py --dataset ogbg-molpcba --gnn gin --filename ginout --emb_dim=256 --epochs=100 --drop_ratio=0.1 --batch_size=512 --lr=0.001
