import argparse
import time

import torch
import torch.nn.functional as F
from torch_geometric import seed_everything

from model import DualGumelGCNConv
from utils import Logger, load_data


class GNN(torch.nn.Module):

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 num_layers,
                 heads=4,
                 K=5,
                 tau=0.25):
        super().__init__()

        Conv = DualGumelGCNConv

        self.convs = torch.nn.ModuleList()
        self.convs.append(Conv(in_dim, hidden_dim, heads, K, tau))
        for _ in range(num_layers - 2):
            self.convs.append(Conv(hidden_dim, hidden_dim, heads, K, tau))
        self.convs.append(Conv(hidden_dim, out_dim, heads, K, tau))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x):
        h = self.convs[0](x)
        h = F.elu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        for _, conv in enumerate(self.convs[1:-1]):
            h = conv(h)
            h = F.elu(h)
            h = F.dropout(h, p=0.5, training=self.training)
        h = self.convs[-1](h)
        return h.log_softmax(dim=-1)


def train(model, data, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(data.x)
    out = out[data.train_mask]
    loss = F.nll_loss(out, data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data):
    model.eval()

    out = model(data.x)

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Chameleon')
    parser.add_argument('--split_type', type=str, default='random')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--heads', type=int, default=1)
    parser.add_argument('--tau', type=int, default=0.25)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    seed_everything(42)
    torch.backends.cudnn.deterministic = True

    dataset, data = load_data(args.dataset, split_type=args.split_type)
    data = data.to(device)

    logger = Logger(args.runs, args)

    for run in range(args.runs):
        time_list = []
        model = GNN(in_dim=data.num_features,
                    hidden_dim=args.hidden_dim,
                    out_dim=dataset.num_classes,
                    num_layers=args.num_layers,
                    heads=args.heads,
                    K=args.K,
                    tau=args.tau).to(device)
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            torch.cuda.synchronize()
            since = time.time()
            loss = train(model, data, optimizer)
            torch.cuda.synchronize()
            time_elapsed = time.time() - since
            time_list.append(time_elapsed)
            result = test(model, data)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')
        time_list = torch.tensor(time_list)
        time_avg = torch.mean(time_list)
        print('Average Training Time is {:.4f}s'.format(time_avg))
        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == '__main__':
    main()
