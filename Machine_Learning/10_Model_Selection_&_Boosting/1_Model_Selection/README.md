# Model Selection

## k-Fold Cross Validation

We use this to find the best accuracy. It makes k different train sets and test sets and trains on each train set and tests it on each test set to find the best accuracy.

### Practical

#### Applying k-Fold Cross Validation

```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))
```

## Grid Search

We use this to find the best hyperparameters.

### Practical

#### Applying Grid Search to find the best model and the best parameters

```python
from sklearn.model_selection import GridSearchCV
parameters = [{"C": [0.25, 0.5, 0.75, 1], "kernel": ["linear"]},
              {"C": [0.25, 0.5, 0.75, 1], "kernel": ["rbf"], 
               "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, 
                           scoring="accuracy", cv=10, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
print("Best Parameters:", best_parameters)
```