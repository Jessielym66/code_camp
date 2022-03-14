import argparse
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GATConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dp):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True))
        for i in range(n_layers - 1):
            self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True))
        self.layers.append(GCNConv(hidden_channels, out_channels, add_self_loops=True, normalize=True))
        self.dropout = nn.Dropout(p=dp)

    def forward(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index)
            h = h.relu()
        return h

class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, dp):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_channels, hidden_channels, add_self_loops=True, normalize=True))
        for i in range(n_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels, add_self_loops=True, normalize=True))
        self.layers.append(GCNConv(hidden_channels, out_channels, add_self_loops=True, normalize=True))
        self.dropout = nn.Dropout(p=dp)

    def forward(self, x, edge_index):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index)
            h = h.relu()
        return h


def train_one_run(args, dataset, device):
    data = dataset[0].to(device)
    
    if args.model_name == 'GCN':
        model = GCN(dataset.num_features, args.hidden, dataset.num_classes, args.layers, args.dropout)
    elif args.model_name == 'GAT':
        model = GAT(dataset.num_features, args.hidden, dataset.num_classes, args.layers, args.dropout)
    model = model.to(device)
    # print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    ###### train ######
    for epoch in range(1, args.epochs+1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        # print("## epoch {}: loss {:.4f}".format(epoch, loss.cpu().detach()))

    ###### valid ######
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim = 1)
    correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = correct.sum() / data.val_mask.sum()
    
    ###### test ######
    model.eval()
    logits = model(data.x, data.edge_index)
    pred = logits.argmax(dim = 1)
    correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = correct.sum() / data.test_mask.sum()
    
    return val_acc, test_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='Cora', name='Cora')
    '''
    print("in:{},  out:{}".format(dataset.num_features, dataset.num_classes))
    data = dataset[0]
    print("data:", data)
    print("number of nodes:", data.num_nodes)        # 2708
    print("number of edges:", data.num_edges)        # 10556
    print("training nodes:", data.train_mask.sum())  # 140
    print("validate nodes:", data.val_mask.sum())    # 
    print("testing nodes:", data.test_mask.sum())    # 
    print("train mask:", data.train_mask)            # [2708]
    print("test mask:", data.test_mask)              # [2708] True or False
    print("data.x:", data.x.shape, "\n    ", data.x) # [2708, 1433]
    print("data.y:", data.y.shape, "\n    ", data.y) # [2708]
    '''

    acc = []
    for run in range(1, args.runs+1):
        val_acc, test_acc = train_one_run(args, dataset, device)
        print("## run {}: val acc = {:.4f},  test acc = {:.4f}".format(run, val_acc, test_acc))
        acc.append([val_acc, test_acc])
    acc = torch.tensor(acc, dtype=float)
    print("## Average val acc: {:.4f},  Average test acc: {:.4f}".format(acc.mean(dim=0)[0], acc.mean(dim=0)[1]))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--model_name", type=str, default="GCN", help="Model name ('GCN', 'GAT')")
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--runs", type=int, default=10, help="training times")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--self_loop", action='store_true', help="graph self-loop (default=False)")
    args = parser.parse_args()

    main(args)