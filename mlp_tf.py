import numpy as np
from six.moves import cPickle as pickle
import tensorflow as tf

image_size = 28
hidden1_units = 1024
num_labels = 10

train_subset = 10000
num_steps = 3001
batch_size = 128

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
    # Map 0 to [1,0,0,...], 1 to [0,1,0,...]
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

def accuracy(predictions, labels):
    return (100.0 *
            np.sum(np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)) /
            predictions.shape[0])

if __name__ == "__main__":
    pickle_file = 'notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

    # Reformat datasets and labels
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

    graph = tf.Graph()
    with graph.as_default():
        # Input dataset
        tf_train_dataset = tf.placeholder(tf.float32,
                                          shape=(batch_size, image_size * image_size))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        # Variables
        weights1 = tf.Variable(
            tf.truncated_normal([image_size * image_size, hidden1_units]))
        biases1 = tf.Variable(tf.zeros([hidden1_units]))
        hidden1 = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)

        weights2 = tf.Variable(
            tf.truncated_normal([hidden1_units, num_labels]))
        biases2 = tf.Variable(tf.zeros([num_labels]))

        def feedforward(dataset):
            h1 = tf.nn.relu(tf.matmul(dataset, weights1) + biases1)
            return tf.matmul(h1, weights2) + biases2

        logits = feedforward(tf_train_dataset)

        loss = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

        # Predictions for the training, validation, and test data
        train_prediction = tf.nn.softmax(logits)
        valid_prediction = tf.nn.softmax(feedforward(tf_valid_dataset))
        test_prediction = tf.nn.softmax(feedforward(tf_test_dataset))

    with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        print('Initiallized')

        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

            # Generate a minibatch
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data,
                         tf_train_labels: batch_labels}

            _, l, predictions = session.run([optimizer, loss, train_prediction],
                                            feed_dict=feed_dict)
            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('\tMinibatch accuracy: %.1f%%' % accuracy(
                      predictions, batch_labels))
                print('\tValidation accuracy: %.1f%%' % accuracy(
                      valid_prediction.eval(), valid_labels))
        print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
