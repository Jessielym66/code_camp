import os
import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
from optuna.trial import TrialState
from optuna.visualization import plot_contour, plot_optimization_history, plot_intermediate_values, plot_parallel_coordinate, plot_slice, plot_param_importances

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load Cora dataset
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
g = g.to(device)

# add self loop
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

# get data
node_features = g.ndata['feat']         # [2708, 1433]
labels = g.ndata['label']               # [2708]
train_mask = g.ndata['train_mask']      # [2708]
val_mask = g.ndata['val_mask']          # [2708]
test_mask = g.ndata['test_mask']        # [2708]

in_feat = g.ndata['feat'].shape[1]
out_feat = dataset.num_classes


# define obj: max  val_acc = model(graph)
def objective(trial):
    #n_layers = trial.suggest_int("n_layers", 2, 3)      ## 2
    hidden = trial.suggest_int("hidden_feat", 15, 20)   ## 16
    dp = trial.suggest_float("dropout", 0.2, 0.5)           ## 0.5
    n_heads = trial.suggest_int("n_heads", 1,5)
    #model = GCN(in_feat, hidden, out_feat, n_layers, dp).to(device)
    model = GCN_GAT(g, in_feat, hidden, out_feat, n_heads, dp).to(device)

    lr = 0.01
    decay = 0.0001
    #lr = trial.suggest_float("lr", 0.001, 0.1, log=True)            ## 0.01
    #decay = trial.suggest_float("decay", 0.00001, 0.001, log=True)  ## 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    
    for epoch in range(1, 200):
        model.train()
        logits = model(g, node_features)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])  #[140,7] [140]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            logits = model(g, node_features)    # [2708,7]
            pred = logits.argmax(1)             # [2708]
            val_acc = (pred[val_mask] == labels[val_mask]).float()  # [500]
            val_acc = val_acc.mean()                                # float num

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()  
    return val_acc


# optimize
study = optuna.create_study(direction = "maximize")
study.optimize(objective, n_trials = 100)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
best_trial = study.best_trial

print("  Value: ", best_trial.value)

print("  Params: ")
for key, value in best_trial.params.items():
    print("    {}: {}".format(key, value))

'''
plot_contour(study)
plot_contour(study, params=['lr', 'n_layers'])
plot_optimization_history(study)
plot_intermediate_values(study)
plot_parallel_coordinate(study)
plot_parallel_coordinate(study, params=['lr', 'n_layers'])
plot_slice(study)
plot_slice(study, params=['lr', 'n_layers'])
plot_param_importances(study)
'''
