# Self Organizing Maps

## Intuition

### How Do Self-Organizing Maps Work?

SOMs are used for reducing dimensions. When we put we have a lot of columns and rows, we can not visualize different groups (clusters). SOMs help us to visualize them on a 2D space. Like this:

![chart](chart-min.PNG)

### How Do Self-Organizing Maps Learn?

The network of SOMs look like this:

![chart2](chart2-min.PNG)

The above picture clearly shows that the output is a 2D map, but to match the previous versions of neural networks, let's look at it from another angle:

![chart3](chart3-min.PNG)

Note that SOMs are completely different from neural netwoks, even if the pictures look alike.

We have weights here like the neural networks, but the difference is that the weights are characteristics of the node not the connection. So, it looks like this:

![chart4](chart4-min.PNG)

On each row of data we calculate the distance between the input layer and each output node with this formula:

![formula](formula.png)

Then, call the node with minimum distance the **best matching unit (BMU)**. In the next step, we update the weights of this node to make it even better. Also, the nodes around this BMU are going to update as well. The closer they to the BMU the more they are going to update. As an illustration look at this image:

![chart5](chart5-min.PNG)

On each epoch, the radius of each BMU becomes smaller. So, the updating becomes more focused.

Important to know:

* SOMs retain topology of the input set
* SOMs reveal correlations that are not easily identified
* SOMs classify data without supervision
* No target vector -> no backpropagation
* No lateral connections between output nodes

## Practical

### Part 1 - SOM

#### Install MiniSom Package

```python
pip install MiniSom
```

#### Training the SOM

```python
from minisom import MiniSom
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate =0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
```

#### Visualizing the results

```python
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markeredgecolor=colors[y[i]], markerfacecolor='None',
         markersize=10, markeredgewidth=2)
show()
```

![train-vis](train-vis.png)

#### Finding the frauds

```python
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(2, 5)], mappings[(1,7)]), axis=0)
frauds = sc.inverse_transform(frauds)
```

#### Printing the Fraunch Clients

```python
print('Fraud Customer IDs')
for i in frauds[:, 0]:
  print(int(i))
```

### Part 2 - Going from Unsupervised to Supervised Deep Learning

#### Create Matrix of Features

```python
customers = dataset.iloc[:, 1:].values
```

#### Create Dependent Variable

```python
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i] = 1
```

### Part 3 - ANN

#### Training the ANN on the Training set

```python
ann.fit(customers, is_fraud, batch_size=1, epochs=10)
```

#### Predicting test set results

```python
y_pred = ann.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
y_pred = y_pred[y_pred[:, 1].argsort()]
```