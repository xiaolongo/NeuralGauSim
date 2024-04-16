import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric import seed_everything
from torch_geometric.nn import GCNConv

from model import DualGumbelAnchorGCNConv
from utils import Logger, load_data


class GNN(torch.nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 num_nodes,
                 anchor_nodes,
                 heads=4,
                 K=5,
                 tau=0.25,
                 add_edge_index=False):
        super().__init__()

        Conv = DualGumbelAnchorGCNConv

        self.add_edge_index = add_edge_index

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            Conv(in_dim, hidden_dim, num_nodes, anchor_nodes, heads, K, tau))
        for _ in range(num_layers - 2):
            self.convs.append(
                Conv(hidden_dim, hidden_dim, num_nodes, anchor_nodes, heads, K,
                     tau))
        self.convs.append(
            Conv(hidden_dim, out_dim, num_nodes, anchor_nodes, heads, K, tau))
        if add_edge_index:
            self.gconvs = torch.nn.ModuleList()
            self.gconvs.append(GCNConv(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.gconvs.append(GCNConv(hidden_dim, hidden_dim))
            self.gconvs.append(GCNConv(hidden_dim, out_dim))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.add_edge_index:
            for gconv in self.gconvs:
                gconv.reset_parameters()

    def forward(self, x, edge_index):
        if self.add_edge_index:
            h = 0.5 * (self.convs[0](x) + self.gconvs[0](x, edge_index))
            h = F.elu(h)
            h = F.dropout(h, p=0.5, training=self.training)
            for i, conv in enumerate(self.convs[1:-1]):
                h = 0.5 * (conv(h) + self.gconvs[i](h, edge_index))
                h = F.elu(h)
                h = F.dropout(h, p=0.5, training=self.training)
            h = 0.5 * (self.convs[-1](h) + self.gconvs[-1](h, edge_index))
        else:
            h = self.convs[0](x)
            h = F.elu(h)
            h = F.dropout(h, p=0.5, training=self.training)
            for _, conv in enumerate(self.convs[1:-1]):
                h = conv(h)
                h = F.elu(h)
                h = F.dropout(h, p=0.5, training=self.training)
            h = self.convs[-1](h)
        return h.log_softmax(dim=-1)


def train(model, data, optimizer, scheduler, adjust_LR=True):
    model.train()

    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    out = out[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                   max_norm=20,
                                   norm_type=2)
    optimizer.step()
    if adjust_LR == 1:
        scheduler.step()
    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()

    out = model(data.x, data.edge_index)

    train_pred = out[data.train_mask].argmax(1)
    train_acc = train_pred.eq(
        data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()

    val_pred = out[data.val_mask].argmax(1)
    val_acc = val_pred.eq(
        data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()

    test_pred = out[data.test_mask].argmax(1)
    test_acc = test_pred.eq(
        data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()

    return train_acc, val_acc, test_acc


@torch.no_grad()
def test_attention(model, data):
    model.eval()
    _, attention_1, attention_2 = model(data.x)
    np.savetxt('attention_1.csv',
               attention_1.cpu().detach().numpy(),
               fmt='%d',
               delimiter=',')
    np.savetxt('attention_2.csv',
               attention_2.cpu().detach().numpy(),
               fmt='%d',
               delimiter=',')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--anchor_nodes', type=int, default=500)
    parser.add_argument('--tau', type=float, default=0.25)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--adjust_LR', type=int, default=0)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    seed_everything(42)

    dataset, data = load_data(args.dataset, split_type=args.split_type)
    data = data.to(device)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        time_list = []
        model = GNN(in_dim=data.num_features,
                    hidden_dim=args.hidden_dim,
                    out_dim=dataset.num_classes,
                    num_layers=args.num_layers,
                    num_nodes=data.num_nodes,
                    anchor_nodes=args.anchor_nodes,
                    heads=args.heads,
                    K=args.K,
                    tau=args.tau).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=0.01,
                                                        pct_start=0.5,
                                                        total_steps=500,
                                                        div_factor=10,
                                                        final_div_factor=10)
        for epoch in range(1, 1 + args.epochs):
            torch.cuda.synchronize()
            since = time.time()
            loss = train(model,
                         data,
                         optimizer,
                         scheduler,
                         adjust_LR=args.adjust_LR)
            torch.cuda.synchronize()
            time_elapsed = time.time() - since
            time_list.append(time_elapsed)
            result = test(model, data)
            logger.add_result(run, result)

            if args.log_steps == 1:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        # test_attention(model, data)
        if args.log_steps == 1:
            time_list = torch.tensor(time_list)
            time_avg = torch.mean(time_list)
            print('Average Training Time is {:.4f}s'.format(time_avg))
            logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()
