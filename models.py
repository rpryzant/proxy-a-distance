"""


"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import random
import time
from sklearn import svm


# class (object):
#     def __init__(self, config):
#         if config.l2 > 0:
#             self.model = svm.LinearSVC(penalty='l2', C=config.l2, probability=True)
#         elif config.l1 > 0:
#             self.model = svm.LinearSVC(penalty='l1', C=config.l1, probability=True)
#         else:
#             self.model = svm.SVC(probability=True) # still regularizes

#     def fit_and_predict(self, data, val_data, sess):
#         x_train, y_train, _ = zip(*data)
#         x_train = [np.reshape(x, [-1]).tolist() for x in x_train] # concatenate errythang
#         self.model.fit(x_train, y_train)

#         x_val, y_val, _ = zip(*val_data)
#         x_val = [np.reshape(x, [-1]).tolist() for x in x_val] # concatenate errythang
#         y_hat = self.model.predict(x_val)
#         y_probs = [x[1] for x in self.model.predict_proba(x_val)]
#         return y_probs, y_hat, 1


class SVM(object):
    def __init__(self, batch_size, vocab_size, max_seq_len=50, hidden_units=128, learning_rate=0.006):
        self.vocab_size = vocab_size
        self.model = svm.SVC(C=30000, probability=True)

    def make_sparse(self, sequence):
        out = [0] * self.vocab_size
        for x in sequence:
            if x == 0:
                break
            out[x] = 1
        return out

    def prepare_examples(self, x, y):
        return [self.make_sparse(xi) + self.make_sparse(yi) for xi, yi in zip(x, y)]

    def prepare_labels(self, d):
        return [np.argmax(di) for di in d]

    def train_on_batch(self, domains, x, x_lens, y, y_lens):
        examples = self.prepare_examples(x, y)
        labels = self.prepare_labels(domains)
        self.model.fit(examples, labels)

    def predict(self, x, x_lens, y, y_lens):
        examples = self.prepare_examples(x, y)
        y_hat = self.model.predict(examples)
        return y_hat

    def fit(self, dataset):
        # TODO
        dataset.set_batch_size(200)#dataset.get_n('train') - 1)
        all_data = next(dataset.mixed_batch_iter())
        self.train_on_batch(*all_data)

    def test(self, dataset):
        dataset.set_batch_size(50)
        test_data = next(dataset.mixed_batch_iter(data='test'))
        print np.array([np.argmax(x) for x in test_data[0]])
        return self.predict(*test_data[1:])


class NN(object):
    def __init__(self, batch_size, vocab_size, max_seq_len=50, hidden_units=128, learning_rate=0.006):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.hidden_units = hidden_units
        self.lr = learning_rate

        self.x = tf.placeholder(tf.int32, [batch_size, max_seq_len])
        self.x_len = tf.placeholder(tf.int32, [batch_size])
        self.y = tf.placeholder(tf.int32, [batch_size, max_seq_len])
        self.y_len = tf.placeholder(tf.int32, [batch_size])
        self.d = tf.placeholder(tf.int32, [batch_size, 2])

        self.E = tf.get_variable(name='E',
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 shape=[vocab_size, hidden_units])
        self.V = tf.get_variable(name='V',
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 shape=[hidden_units, 2])

        input = tf.nn.embedding_lookup(self.E, self.x)
        input = tf.reshape(input, [batch_size, -1])
        with tf.variable_scope('fc1'):
            fc1 = self.fc(input, self.max_seq_len * self.hidden_units, 1024)
        with tf.variable_scope('fc2'):
            fc2 = self.fc(fc1, 1024, 2)
        self.logits = fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.d))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())


    def fc(self, input, in_dim, out_dim):
        W = tf.get_variable(name='W1',
                             initializer=tf.contrib.layers.xavier_initializer(),
                             shape=[in_dim, out_dim])
        b = tf.get_variable(name='b',
                             initializer=tf.contrib.layers.xavier_initializer(),
                             shape=[out_dim])
        return tf.matmul(input, W) + b


    def train_on_batch(self, domains, x, x_lens, y, y_lens):
        _, loss, preds = self.sess.run([self.train_step, self.loss, self.logits],
                                feed_dict={
                                    self.x: x,
                                    self.x_len: x_lens,
                                    self.y: y,
                                    self.y_len: y_lens,
                                    self.d: domains
                                    })

        # print [np.argmax(x) for x in domains]
        # print [np.argmax(x) for x in preds]
        # print loss
        return loss, preds

if __name__ == '__main__':
    d = DomainClassifier(5, 30000)
