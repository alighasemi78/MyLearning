# Regression Model Selection in Python

We have to train all of our models on our dataset, and then calculate the R^2^ scores and then compare the models and choose the model with the best score.

## Evalueating the Model Performance

```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```