# Natural Language Processing

## Intuition

### Types of NLP

* NLP
* DNLP: combination of NLP and deep learning
* Seq2Seq: a subset of DNLP

### Classical vs Deep Learning Models

Some examples:

1. If / Else rules (Chatbot) **(NLP)**
2. Audio frequency components analysis (Speech Recognition) **(NLP)**
3. Bag-of-words model (Classification) **(NLP)**
4. CNN for text recognition (Classification) **(DNLP)**
5. Seq2Seq (many applications) **(Seq2Seq)**

### Bag-Of-Words Model

In this mode, we convert the collection of words to a vector. Native English speakers usually use 20000 words. So, we could make a 20000 elements long vector. Note that the first element is called **SOS** (Start Of Senctence) and the second element is called **EOS** (End Of Sentence). The last element is also there to store special words. Words that we did not consider in our vector. Now we go through our sentences and add a 1 to each element of vector that we see its corresponding word. Then, we gather data for training. We want to answer YES or NO. By gathering data and the results of that data we can use for example logistic regression to predict the answer.

## Practical

### Importing the dataset

```python
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
```

### Cleaning the texts

```python
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(len(dataset.values)):
    review = re.sub('[^a-zA-Z]', " ", dataset.values[i][0])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words("english")
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = " ".join(review)
    corpus.append(review)
```

### Creating the Bag of Words model

```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
```