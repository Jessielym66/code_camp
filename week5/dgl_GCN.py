import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import tensorboard
import argparse

class GCN(nn.Module):
    def __init__(self, in_features, h_features, num_classes, n_layers, dp):
        super(GCN, self).__init__()
        self.dropout = nn.Dropout(p=dp)
        self.layers = nn.ModuleList()
        self.layers.append(GraphConv(in_features, h_features, activation=F.relu))
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(h_features, h_features, activation = F.relu))
        self.layers.append(GraphConv(h_features, num_classes))

    def forward(self, g, x):
        h = x
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)
    
    f = "lr{}-dp{}-decay{}".format(args.lr, args.dropout, args.weight_decay)
    os.mkdir(os.path.join(args.logs, f))
    log_dir = os.path.join(args.logs, f)
    
    # load Cora dataset
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    g = g.to(device)
    
    # add self loop
    if args.add_self_loop:
        g = dgl.remove_self_loop(g)
        g = dgl.add_self_loop(g)
        n_edges = g.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    norm = norm.to(dev)
    g.ndata['norm'] = norm.unsqueeze(1)
    
    node_features = g.ndata['feat']         # [2708, 1433]
    labels = g.ndata['label']               # [2708]
    train_mask = g.ndata['train_mask']      # [2708]
    val_mask = g.ndata['val_mask']          # [2708]
    
    ############# train #############
    acc = []
    for i in range(args.runs):
        model = GCN(g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, args.layers, args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        if args.runs > 1:
            os.mkdir(os.path.join(args.logs, f, 'run_'+str(i)))
            log_dir = os.path.join(args.logs, f, 'run_'+str(i))
        writer = tensorboard.SummaryWriter(log_dir=log_dir)
        step = 0
            
        for epoch in range(1, args.epochs+1):
            model.train()
            logits = model(g, node_features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])  #[140,7] [140]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('train_loss', loss.data.cpu().numpy(), global_step=step)
            step += 1
            
            model.eval()
            with torch.no_grad():
                logits = model(g, node_features)    # [2708,7]
                pred = logits.argmax(1)             # [2708]
                val_acc = (pred[val_mask] == labels[val_mask]).float()  # [500]
                val_acc = val_acc.mean()                                # float num
                writer.add_scalar('val_acc', val_acc, global_step=epoch)
                #print("Epoch {}\t train_loss: {:.3f}\t val_acc: {:.3f}%".format(epoch, loss, val_acc*100))
                
        ############# test #############
        test_mask = g.ndata['test_mask']
        model.eval()
        logits = model(g, node_features)
        pred = logits.argmax(1)
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        acc.append(test_acc)
        print("# run {}: acc {:.4f}".format(i, test_acc))
        
    acc = torch.tensor(acc, dtype=float)
    print("average acc: {:.4f}".format(acc.mean()))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'PyTorch GCN')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--logs', type=str, default='./logs', help='the path to save model and results')
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--runs", type=int, default=1, help="training times")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--add_self_loop", action='store_true', help="graph self-loop (default=False)")
    args = parser.parse_args()
    
    main(args)