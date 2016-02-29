import os
import pickle
from sklearn.linear_model import LogisticRegression

# accuracy
# num of training
# 50     => 0.509
# 100    => 0.697
# 1000   => 0.833
# 5000   => 0.851
# 200000 => 0.893

if __name__ == '__main__':
    pickle_file = 'notMNIST.pickle'

    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    train_dataset = data['train_dataset']
    train_labels = data['train_labels']
    test_dataset = data['test_dataset']
    test_labels = data['test_labels']

    num_train = len(train_dataset)
    num_test = len(test_dataset)

    print("num_train:", num_train)
    print("num_test:", num_test)

    train_dataset = train_dataset.reshape(num_train, -1)
    test_dataset = test_dataset.reshape(num_test, -1)

    # Logistic Regression
    max_train = 5000
    model_file = "logreg_%d.pickle" % max_train
    if not os.path.exists(model_file):
        logreg = LogisticRegression()
        logreg.fit(train_dataset[:max_train], train_labels[:max_train])
        with open(model_file, "wb") as f:
            pickle.dump(logreg, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(model_file, "rb") as f:
            logreg = pickle.load(f)

    print("accuracy:", logreg.score(test_dataset, test_labels))
