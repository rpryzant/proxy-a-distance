"""
SVM for Domain discrimination
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
        """ Convert sequence of word-indices into one-hot vector
        """
        out = [0] * self.vocab_size
        for x in sequence:
            if x == 0:
                break
            out[x] = 1
        return out


    def prepare_examples(self, x, xl, y, yl):
        """ Convert examples into a sparse matrix of one-hot vectors
            each vector is the concatination of x (sequence from corpus 1) and 
            y (sequence from corpus 2)
        """
        return csr_matrix([self.vectorize(xi[:xli] + yi[:yli]) \
                               for xi, xli, yi, yli in zip(x, xl, y, yl)])


    def prepare_labels(self, d):
        """ Translate probability distributions to binary label
        """
        return [np.argmax(di) for di in d]


    def train_on_batch(self, domains, x, x_lens, y, y_lens, c=3000):
        """ Train svm on some data
        """
        self.model = svm.SVC(C=c, probability=True)
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        labels = self.prepare_labels(domains)
        self.model.fit(examples, labels)


    def predict(self, x, x_lens, y, y_lens):
        """ Predict domains for some examples
        """
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        y_hat = self.model.predict(examples)
        return y_hat


    def mse(self, labels, x, x_lens, y, y_lens):
        """ mean squared error (MSE)
        """
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        y_hat = self.model.predict_proba(examples)
        mse = mean_squared_error(labels, y_hat)
        return mse


    def mae(self, labels, x, x_lens, y, y_lens):
        """ mean absolute error (MAE)
        """
        examples = self.prepare_examples(x, x_lens, y, y_lens)
        y_hat = self.model.predict_proba(examples)
        mae = mean_absolute_error(labels, y_hat)
        return mae


    def fit(self, dataset):
        """ Fit the model to a dataset and evaluate with MAE
        """
        dataset.set_batch_size(dataset.get_n('train') - 1)
        train_data = next(dataset.mixed_batch_iter())

        dataset.set_batch_size(dataset.get_n('val') - 1)
        val_data = next(dataset.mixed_batch_iter(data='val'))
        print 'INFO: training on ', dataset.get_n('train') - 1, ' examples'
        self.train_on_batch(*train_data)  
        mae = self.mae(*val_data)
        print 'INFO: mae: ', mae


    def test(self, dataset, mae=False):
        """ Test the dataset on a dataset
        """
        dataset.set_batch_size(dataset.get_n('test') - 1)
        test_data = next(dataset.mixed_batch_iter(data='test'))
        if not mae:
            return self.mse(*test_data)
        else:
            return self.mae(*test_data)

