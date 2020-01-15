import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

remove = ('headers', 'footers', 'quotes')

print("Loading 20 newsgroups dataset for categories:")

train_data = fetch_20newsgroups(subset='train', shuffle=True, remove=remove)
test_data = fetch_20newsgroups(subset='test', shuffle=True, remove=remove)
print('data loaded')

categories = train_data.target_names

train_target = train_data.target
test_target = test_data.target

vectorizer = text.TfidfVectorizer(max_features=600)
input_train = vectorizer.fit_transform(train_data.data).toarray()
input_test = vectorizer.transform(test_data.data).toarray()

input_dim, hidden_dim, out_dim = 600, 100, 20
w1 = np.random.rand(input_dim, out_dim)
learning_rate = 2e-1
epochs = 10

num_correct = 0
result = []
for i in range(epochs):
    num_correct = 0
    for i, news in enumerate(input_train):
        y = np.matmul(news, w1)

        out_index = y.argmax(0)
        out = np.zeros(20)
        out[out_index] = 1

        if out_index == train_target[i]:
            num_correct += 1
        else:
            new_target = np.zeros(20)
            new_target[train_target[i]] = 1
            new_target = new_target.reshape((1, 20))
            news = news.reshape((600, 1))
            w1 += learning_rate * np.matmul(news, new_target)

        result.append([out_index, train_target[i]])
    print("**** Accuracy in train is: ", num_correct / len(input_train))

print("result : ", result)

