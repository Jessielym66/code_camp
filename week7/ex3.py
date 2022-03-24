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
    n_layers = trial.suggest_int("n_layers", 2, 3)      ## 2
    hidden = trial.suggest_int("hidden_feat", 15, 20)   ## 16
    dp = trial.suggest_float("dropout", 0, 1)           ## 0.5
    model = GCN(in_feat, hidden, out_feat, n_layers, dp).to(device)

    lr = trial.suggest_float("lr", 0.001, 0.1, log=True)            ## 0.01
    decay = trial.suggest_float("decay", 0.00001, 0.001, log=True)  ## 0.0005
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
Study statistics:
  Number of finished trials:  100
  Number of pruned trials:  88
  Number of complete trials:  12
Best trial:
  Value:  0.7860000133514404
  Params:
    n_layers: 2
    hidden_feat: 17
    dropout: 0.3976733658029088
    lr: 0.00492024595138278
    decay: 1.0129394227394111e-05
'''

'''
best trial is 13:
Trial 13 finished with value: 0.7860000133514404 and parameters: 
    {'n_layers': 2, 
     'hidden_feat': 17, 
     'dropout': 0.3976733658029088, 
     'lr': 0.00492024595138278, 
     'decay': 1.0129394227394111e-05}. 
Best is trial 13 with value: 0.7860000133514404.
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

