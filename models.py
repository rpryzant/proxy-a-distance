"""


"""
import numpy as np
import sys
import random
import time
from collections import Counter
from sklearn import svm
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



class SVM(object):
    def __init__(self, batch_size, vocab_size, max_seq_len=50, hidden_units=128, learning_rate=0.006):
        self.vocab_size = vocab_size
        self.model = None


    def vectorize(self, sequence):
        out = [0] * self.vocab_size
        for x in sequence:
            if x == 0:
                break
            out[x] = 1
        return out


    def prepare_examples(self, x, xl, y, yl):
        return csr_matrix([self.vectorize(xi[:xli] + yi[:yli]) for xi, xli, yi, yli in zip(x, xl, y, yl)])


    def prepare_labels(self, d):
        return [np.argmax(di) for di in d]


    def train_on_batch(self, domains, x, x_lens, y, y_lens, c=3000):
        self.model = svm.SVC(C=c, probability=True)
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        labels = self.prepare_labels(domains)
        self.model.fit(examples, labels)


    def predict(self, x, x_lens, y, y_lens):
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        y_hat = self.model.predict(examples)
        return y_hat


    def mse(self, labels, x, x_lens, y, y_lens):
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        y_hat = self.model.predict_proba(examples)
        mse = mean_squared_error(labels, y_hat)
        return mse


    def mae(self, labels, x, x_lens, y, y_lens):
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        y_hat = self.model.predict_proba(examples)
        mae = mean_absolute_error(labels, y_hat)
        return mae


    def fit(self, dataset):
        dataset.set_batch_size(dataset.get_n('train') - 1)
        train_data = next(dataset.mixed_batch_iter())

        dataset.set_batch_size(dataset.get_n('val') - 1)
        val_data = next(dataset.mixed_batch_iter(data='val'))
        print 'INFO: training on ', dataset.get_n('train') - 1, ' examples'
        self.train_on_batch(*train_data)  
        mae = self.mae(*val_data)
        print 'INFO: mae: ', mae


    def test(self, dataset, mae=False):
        dataset.set_batch_size(dataset.get_n('test') - 1)
        test_data = next(dataset.mixed_batch_iter(data='test'))
        if not mae:
            return self.mse(*test_data)
        else:
            return self.mae(*test_data)


# class NN(object):
#     """
#     !!!  DEPRECIATED  !!! 
#     """
#     def __init__(self, batch_size, vocab_size, max_seq_len=50, hidden_units=128, learning_rate=0.006):
#         self.batch_size = batch_size
#         self.max_seq_len = max_seq_len
#         self.vocab_size = vocab_size
#         self.hidden_units = hidden_units
#         self.lr = learning_rate

#         self.x = tf.placeholder(tf.int32, [batch_size, max_seq_len])
#         self.x_len = tf.placeholder(tf.int32, [batch_size])
#         self.y = tf.placeholder(tf.int32, [batch_size, max_seq_len])
#         self.y_len = tf.placeholder(tf.int32, [batch_size])
#         self.d = tf.placeholder(tf.int32, [batch_size, 2])

#         self.E = tf.get_variable(name='E',
#                                  initializer=tf.contrib.layers.xavier_initializer(),
#                                  shape=[vocab_size, hidden_units])
#         self.V = tf.get_variable(name='V',
#                                  initializer=tf.contrib.layers.xavier_initializer(),
#                                  shape=[hidden_units, 2])

#         input = tf.nn.embedding_lookup(self.E, self.x)
#         input = tf.reshape(input, [batch_size, -1])
#         with tf.variable_scope('fc1'):
#             fc1 = self.fc(input, self.max_seq_len * self.hidden_units, 1024)
#         with tf.variable_scope('fc2'):
#             fc2 = self.fc(fc1, 1024, 2)
#         self.logits = fc2

#         self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.logits, self.d))

#         optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#         self.train_step = optimizer.minimize(self.loss)

#         self.sess = tf.Session()
#         self.sess.run(tf.global_variables_initializer())


#     def fc(self, input, in_dim, out_dim):
#         W = tf.get_variable(name='W1',
#                              initializer=tf.contrib.layers.xavier_initializer(),
#                              shape=[in_dim, out_dim])
#         b = tf.get_variable(name='b',
#                              initializer=tf.contrib.layers.xavier_initializer(),
#                              shape=[out_dim])
#         return tf.matmul(input, W) + b


#     def train_on_batch(self, domains, x, x_lens, y, y_lens):
#         _, loss, preds = self.sess.run([self.train_step, self.loss, self.logits],
#                                 feed_dict={
#                                     self.x: x,
#                                     self.x_len: x_lens,
#                                     self.y: y,
#                                     self.y_len: y_lens,
#                                     self.d: domains
#                                     })

#         # print [np.argmax(x) for x in domains]
#         # print [np.argmax(x) for x in preds]
#         # print loss
#         return loss, preds

# if __name__ == '__main__':
#     d = DomainClassifier(5, 30000)
