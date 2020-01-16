import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction import text

remove = ('headers', 'footers', 'quotes')

print("Loading 20 newsgroups dataset for categories:")

train_data = fetch_20newsgroups(subset='train', shuffle=True, remove=remove)
test_data = fetch_20newsgroups(subset='test', shuffle=True, remove=remove)
print('data loaded')

categories = train_data.target_names
print(categories)

train_target = train_data.target
test_target = test_data.target

vectorizer = text.TfidfVectorizer(max_features=5000)
input_train = vectorizer.fit_transform(train_data.data).toarray()
input_test = vectorizer.transform(test_data.data).toarray()

input_dim, out_dim = 5000, 20
w1 = np.random.rand(input_dim, out_dim)
learning_rate = 1
epochs = 50

num_correct = 0
result = []
# Trainiing phase
for epoch in range(epochs):
    num_correct = 0
    result = []
    for i, news in enumerate(input_train):
        y = np.matmul(news, w1)

        out_index = y.argmax(0)
        out = np.full((20), -1)
        out[out_index] = 1

        if out_index == train_target[i]:
            num_correct += 1
        else:
            new_target = np.full((20), -1)
            new_target[train_target[i]] = 1
            new_target = new_target.reshape((1, 20))
            news = news.reshape((5000, 1))
            w1 += learning_rate * np.matmul(news, new_target)

        result.append([out_index, train_target[i]])
    print("**** Accuracy of train in epoch", epoch + 1, " is: ", num_correct / len(input_train))

num_correct_test = 0
result_test = []
# Test phase
for i, news in enumerate(input_test):
    y = np.matmul(news, w1)

    out_index = y.argmax(0)

    if out_index == test_target[i]:
        num_correct_test += 1

    result_test.append([out_index, test_target[i]])

print("****** Accuracy of test is: ", num_correct_test / len(input_test))