# Classification Model Selection in Python

We have to train all of our models on our dataset, and then calculate the confusion matrix and the accuracy and then compare the models and choose the model with the best score.

## Evalueating the Model Performance

```python
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
```