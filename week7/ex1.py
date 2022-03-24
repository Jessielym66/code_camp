import optuna

## min (x+1)^2
def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)     # [-10,10]
    return (x + 1) ** 2

## search
study = optuna.create_study()
study.optimize(objective, n_trials = 100)

## results
best_x = study.best_params
best_obj = study.best_value
best_trial = study.best_trial
print("best x:{}".format(best_x))
print("best_obj:{}".format(best_obj))
print("best_trial:{}".format(best_trial))

## continue optimize
#study.optimize(objective, n_trials = 50)
print("num trials:", len(study.trials))     # 100


