import optuna
from optuna.visualization import plot_contour, plot_optimization_history, plot_intermediate_values, plot_parallel_coordinate, plot_slice, plot_param_importances
import scipy.misc

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)     # [-10,10]
    y = trial.suggest_uniform('y', -10, 10)
    return (x + x*y - 2) ** 2

## search
study = optuna.create_study()
study.optimize(objective, n_trials = 20)


## results
best_x = study.best_params
best_obj = study.best_value
best_trial = study.best_trial
'''
plot_contour(study)
plot_contour(study, params=['lr', 'n_layers'])
plot_optimization_history(study)
plot_intermediate_values(study)
plot_parallel_coordinate(study)
plot_parallel_coordinate(study, params=['lr', 'n_layers'])
plot_slice(study)
plot_slice(study, params=['lr', 'n_layers'])
'''
fig = plot_param_importances(study)

scipy.misc.imsave('./ex3/importances', fig)