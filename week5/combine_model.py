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


class GATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2*out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self,edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1) 
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h':h}

    def forward(self, h):
        z = self.fc(h)
        self.g.ndata['z'] = z 
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')

class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim , out_dim , num_heads=1, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge


    def forward(self, h):
        head_out = [attn_head(h) for attn_head in self.heads]
        if self.merge=='cat':
            return torch.cat(head_out, dim=1)
        else:
            return torch.mean(torch.stack(head_out))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim , out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g , in_dim, hidden_dim, num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim*num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


class GCN_GAT(nn.Module):
    def __init__(self, g, in_features, hidden_dim, num_classes, num_heads, dp):
        super(GCN_GAT, self).__init__()
        self.dropout = nn.Dropout(p=dp)
        self.layer1 = GraphConv(in_features, hidden_dim*num_heads)
        self.layer2 = MultiHeadGATLayer(g, hidden_dim*num_heads, num_classes, 1)

    def forward(self, g, x):
        h = self.layer1(g, x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(h)
        return h


class GAT_GCN(nn.Module):
    def __init__(self, g, in_features, hidden_dim, num_classes, num_heads, dp):
        super(GAT_GCN, self).__init__()
        self.dropout = nn.Dropout(p=dp)
        self.layer1 = MultiHeadGATLayer(g, in_features, hidden_dim*num_heads, 1)
        self.layer2 = GraphConv(hidden_dim*num_heads, num_classes)
        
    def forward(self, g, x):
        h = self.layer1(x)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.layer2(g, h)
        return h


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    test_mask = g.ndata['test_mask']
    
    if args.model_name == 'GCN_GAT':
        model = GCN_GAT(g, g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, args.heads, args.dropout).to(device)
        #log_dir = os.path.join('./logs/GCN_GAT/', str(args.heads))
        log_dir = './logs/new/GCN_GAT'
    elif args.model_name == 'GAT_GCN':
        model = GAT_GCN(g, g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, args.heads, args.dropout).to(device)
        #log_dir = os.path.join('./logs/GAT_GCN/', str(args.heads))
        log_dir = './logs/new/GAT_GCN'
    elif args.model_name == 'GAT':
        model = GAT(g, g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, args.heads).to(device)
        #log_dir = os.path.join('./logs/GAT/', str(args.heads))
        log_dir = './logs/new/GAT'
    elif args.model_name == 'GCN':
        model = GCN(g.ndata['feat'].shape[1], args.hidden, dataset.num_classes, args.layers, args.dropout).to(device)
        log_dir = './logs/new/GCN'
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    
    ############# train ############# 
    step = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        logits = model(g, node_features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])  #[140,7] [140]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', loss.data.cpu().numpy(), global_step=step)
        val_loss = F.cross_entropy(logits[val_mask], labels[val_mask])
        writer.add_scalar('val_loss', val_loss.data.cpu().numpy(), global_step=step) 
        step += 1
        
        model.eval()
        with torch.no_grad():
            logits = model(g, node_features)    # [2708,7]
            pred = logits.argmax(1)             # [2708]
            val_acc = (pred[val_mask] == labels[val_mask]).float()  # [500]
            val_acc = val_acc.mean()                                # float num
            writer.add_scalar('val_acc', val_acc, global_step=epoch)
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
            writer.add_scalar('test_acc', val_acc, global_step=epoch)
            #print("Epoch {}\t train_loss: {:.3f}\t val_acc: {:.3f}%".format(epoch, loss, val_acc*100))
            
    ############# test #############
    model.eval()
    logits = model(g, node_features)
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    print("test acc {:.4f}".format(test_acc))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'PyTorch GCN')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--logs', type=str, default='./logs', help='the path to save model and results')
    parser.add_argument("--dropout", type=float, default=0.5, help="dropout probability")
    parser.add_argument("--runs", type=int, default=1, help="training times")
    parser.add_argument("--epochs", type=int, default=200, help="number of training epochs")
    parser.add_argument("--hidden", type=int, default=16, help="number of hidden gcn units")
    parser.add_argument("--layers", type=int, default=1, help="number of hidden gcn layers")
    parser.add_argument("--heads", type=int, default=1, help="number of heads in GAT layer")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")
    parser.add_argument("--add_self_loop", action='store_true', help="graph self-loop (default=False)")
    parser.add_argument("--model_name", type=str, default='GCN_GAT', help="combine GCN and GAT model")
    args = parser.parse_args()
    
    main(args)