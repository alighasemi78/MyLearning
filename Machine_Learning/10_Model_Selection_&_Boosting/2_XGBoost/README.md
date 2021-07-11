# XGBoost

## Practical

### Training XGBoost on the Training set

```python
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```