# Data Preprocessing

## Importing the libraries

The most often libraries used in Machine Learning are:

* [numpy](https://numpy.org/): To make it easier to work with arrays and matrices.
* [matplotlib](https://matplotlib.org/): To plot charts.
* [pandas](https://pandas.pydata.org/): To make it easier to work with the datasets and store them in dataframes.

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

we assign shortcuts to these libraries with the "as" keyword to make it easier to call.

## Importing the dataset

To read the dataset that is in ".csv" format, we use the pandas library as shown below:

```python
dataset = pd.read_csv("Data.csv")
```

Now, we have to seperate the features (independent variables) and the dependent variable. Usually the dependent variable is in the last column and the features are in the first columns. To seperate the features and dependent variable, we have to do this:

```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

Here, we can use "iloc" because when we read the ".csv" file, pandas converted the dataset to a dataframe. What "iloc" does is that it extracts a part of table. As shown above it takes 2 inputs. The first one is for selecting which rows we want to extract, and the second one is for columns. For the features and dependent variable we want to extracts all the rows. So, we only put a ":". ":" specifies a range. The left part of ":" specifies the min and the right part specifies the max of the range. When the left part is not specified, python infers that part as the minimum and when the right part is not specified, python infers that part as the maximum. In the code above, we have used -1 as an index. In python, -1 means the last index. So, ":-1" means from min up until the last index (Python includes the start of range and excludes the end of range). Moreover, "-1" means only the last index. We use the ".values" to tell pandas that we want the values of the specified rows and columns.

## Taking care of missing data

Generally we do not want missing data in our dataset because it can cause error when training our Machine Learning model. There are several ways to handle them:

* Ignore the observation. Works if the number of ignored data is under 1 percent of the whole data.
* Replace the missing data by the average all of the data in the same column.

We use the second way. To do this we use a library called [scikit-learn](https://scikit-learn.org/) which is a very usefull library in Machine Learning.

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])
```

In the code above we instantiate the class "SimpleImputer" with two option. The "missing_value" specifies what cells to replace. We chose "np.nan" because we want to replace cells that are empty. The "strategy" specifies by which value will the empty cell be replaced. We chose "mean" because we want to replace empty cells by the average of all the data in the same columns. Then we call the fit method on the features that have empty columns and are numerical. This function calculates the mean values. Then we call the transform method which returns the columns that had empty values with the new values.

## Encoding categorical data

It is hard to make correlations between the categorical columns and the dependent variable. As a result, we need to convert these columns to numerical. At first, you might think we could just transform them to numbers begining from 1 to the number of categories. However, the future Machine Learning model will think that there are differences between these categories corresponding their values. Therefore, we use **One Hot Encoding**.

### Encoding the Independent Variable

For the variables that have more than 2 categories, we do as bellow:

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],
    remainder="passthrough")
X = np.array(ct.fit_transform(X=X))
```

In the "ColumnTransformer" we have 2 options. The "transformers" option takes 3 arguments. The first one specifies that we are going to encode our variable to One Hot format. The second argument sepcifies the class that is going to do the encoding. The third arguments takes the indexes of the columns we want to encode. Then, we call the fit and transform method to run the encoding and replace the categorical columns with the new columns. Pay attention that we must store numpy arrays in X and y, that is why we call the "np.array" on the returned list by the "fit_transform" method as it is not one.

### Encoding the Dependent Variable

For the variables that have only 2 categories, we do as bellow:

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y=y)
```

The "LabelEncoder" encodes the values of the variable to 0s and 1s.

## Splitting the dataset into the Training set and Test set

Pay attention that we have to apply feature scaling before splitting the dataset into the training set and Test set. That is because we do not want the scaling to affect the test set. The test set is something we should not work with and has to be something new to our training procedure.
To Split the dataset we do as bellow:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
```

The "test_size" option specifies what proportion of the data becomes the test set.

## Feature Scaling

In to prevent some features dominate other features in some Machine Learning models, we need feature scaling. We do not need to do feature scaling in all ML models.
The main 2 feature scaling methos are:

* Standardisation

  $x_{stand}=\frac{x-mean(x)}{sd^*(x)}$
  x_{stand}=\frac{x-mean(x)}{sd^*(x)}

  sd = standard deviation

* Normalisation

  $x_{norm}=\frac{x-min(x)}{max(x)-min(x)}$

Normalization converts numbers to the interval of 0 and 1, but standardization converts them to the interval of -3 and 3.
Normalization is recommended if we have a normal distribution, but standardisation works well most of the time. We use standardization as shown bellow:

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X=X_train[:, 3:])
X_test[:, 3:] = sc.transform(X=X_test[:, 3:])
```
