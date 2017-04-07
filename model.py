"""


"""
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sys
import random
import time

class DomainClassifier(object):
    def __init__(self, batch_size, vocab_size, max_seq_len=30, hidden_units=128, num_classes=2, learning_rate=0.003):
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.hidden_units = hidden_units
        self.lr = learning_rate

        self.x = tf.placeholder(tf.int32, [batch_size, max_seq_len])
        self.x_len = tf.placeholder(tf.int32, [batch_size])
        self.y = tf.placeholder(tf.int32, [batch_size, max_seq_len])
        self.y_len = tf.placeholder(tf.int32, [batch_size])
        self.d = tf.placeholder(tf.int32, [batch_size, num_classes])
        self.E = tf.get_variable(name='E',
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 shape=[vocab_size, hidden_units])
        self.V = tf.get_variable(name='V',
                                 initializer=tf.contrib.layers.xavier_initializer(),
                                 shape=[hidden_units*2, num_classes])

        logits = self.forward_pass()
        pred = tf.nn.softmax(logits)
        self.loss = self.calc_loss(logits)
        self.train_step = self.backward_pass(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def backward_pass(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        train_step = optimizer.minimize(loss)
        return train_step


    def calc_loss(self, logits):
        losses = tf.nn.softmax_cross_entropy_with_logits(logits, self.d)
        loss_per_batch = tf.reduce_sum(losses) / tf.to_float(self.batch_size)
        mean_loss = tf.reduce_mean(loss_per_batch)
        return mean_loss
        
    def forward_pass(self):
        with tf.variable_scope('d1'):
            d1_outs, d1_final_state = self.encode(self.x, self.x_len)
        with tf.variable_scope('d2'):
            d2_outs, d2_final_state = self.encode(self.y, self.y_len)

        d1_final = d1_outs[:,-1,:]
        d2_final = d2_outs[:,-1,:]
        final_out = tf.concat(1, [d1_final, d2_final])

        logits = tf.matmul(final_out, self.V)

        return logits


    def encode(self, inputs, lengths, num_classes=2, keep_prob=0.5):
        input_embeddings = tf.nn.embedding_lookup(self.E, inputs)
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=self.hidden_units)
        outputs, state = tf.nn.dynamic_rnn(cell=cell,
                                           inputs=input_embeddings,
                                           sequence_length=lengths,
                                           dtype=tf.float32)
        return outputs, state

        # output layer
        V = tf.get_variable(
            name='V',
            initializer=tf.random_normal_initializer(),
            shape=[hidden_units, num_classes])


    def train_on_batch(self, x, x_lens, y, y_lens, domains):
        _, loss = self.sess.run([self.train_step, self.loss],
                                feed_dict={
                                    self.x: x,
                                    self.x_lens: x_lens,
                                    self.y: y,
                                    self.y_lens: y_lens,
                                    self.d: domains
                                    })
        return loss

if __name__ == '__main__':
    d = DomainClassifier(5, 30000)
