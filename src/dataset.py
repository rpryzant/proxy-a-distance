"""
Utility class for managing datasets
"""
import numpy as np
import random


class Dataset(object):
    
    def __init__(self, d1_source, d1_target, d2_source, d2_target, vocab, batch_size=64, max_seq_len=50):
        #   {word : id} mapping
        self.vocab_map = self.build_vocab_mapping(vocab) 

        self.d1_source_data = self.prepare_data(d1_source)
        self.d1_target_data = self.prepare_data(d1_target)

        self.d2_source_data = self.prepare_data(d2_source)
        self.d2_target_data = self.prepare_data(d2_target)

        self.train_indices, self.val_indices, self.test_indices = self.make_splits(len(self.d1_source_data))

        self.vocab_size = len(self.vocab_map)
        self.train_n = len(self.train_indices)
        self.val_n = len(self.val_indices)
        self.test_n = len(self.test_indices)

        self.batch_index = 0
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len


    def set_batch_size(self, b):
        """ Set batch size for batch iteration
        """
        self.batch_size = b

    def get_n(self, data='train'):
        """ Get number of examples for some split
        """
        if data == 'train':
            return self.train_n
        elif data == 'val':
            return self.val_n
        elif data == 'test':
            return self.test_n


    def get_vocab_size(self):
        """ Get number of tokens in vocabulary
        """
        return self.vocab_size


    def make_splits(self, N):
        """ Generate train/val/test splits. Train is 7/8 of the data, val/test are each 1/16.
        """
        indices = np.arange(N)
        train_test_n = N / 8

        train = indices[:N - (train_test_n * 2)]
        val = indices[len(train): N - train_test_n]
        test = indices[len(train) + len(val):]

        return train, val, test


    def build_vocab_mapping(self, vocab):
        """ Given a 1-token-per-line vocab file, generate {word : index} mapping.
            Note that the <pad> token is appended to the vocabulary.
        """
        out = {w.split()[0].strip(): i+1 for (i, w) in enumerate(open(vocab))}
        out['<pad>'] = 0
        return out

        
    def prepare_data(self, corpus):
        """ Translate tokens in a 1-sequence-per-line corpus file into
            their mapped indexes, and return a 2d array representation
        """
        dataset = []
        for l in open(corpus):
            dataset.append([self.vocab_map.get(w, self.vocab_map['<unk>']) for w in l.split()])
        return dataset


    def mixed_batch_iter(self, data='train'):
        """ Yields mixed batches of data from both datasets

            A batch is
              ([x in corpus 1], [len(x)], [y in corpus 2], [len(y)])
        """
        if data == 'train':
            indices = self.train_indices
        elif data == 'val':
            indices = self.val_indices
        elif data == 'test':
            indices = self.test_indices

        while self.has_next_batch(indices):
            out_batch = ([], [], [], [], [])
            for i in range(self.batch_size):
                j = indices[self.batch_index + i]
                if j >= min(len(self.d1_source_data), len(self.d1_target_data),
                            len(self.d2_source_data), len(self.d2_target_data)):
                    break
                # mixing probability = 0.5
                if random.random() < 0.5:
                    x, x_l, y, y_l = self.get_example(self.d1_source_data, self.d1_target_data, j)
                    domain = [1, 0]
                else:
                    x, x_l, y, y_l = self.get_example(self.d2_source_data, self.d2_target_data, j)
                    domain = [0, 1]

                out_batch[0].append(domain)
                out_batch[1].append(x)
                out_batch[2].append(x_l)
                out_batch[3].append(y)
                out_batch[4].append(y_l)

            yield out_batch
            self.batch_index += self.batch_size

        self.batch_index = 0
            

    def has_next_batch(self, indices):
        """ tests whether another batch can be expelled
        """
        return self.batch_index + self.batch_size < len(indices)


    def get_example(self, source, target, i):
        """ Generates a single pair of padded examples from the data
        """
        def post_pad(x, pad=0):
            new =  [pad] * self.max_seq_len
            new[:len(x)] = x
            return new[:self.max_seq_len]

        x = source[i]
        x = post_pad(x)
        x_l = np.count_nonzero(x)

        y = target[i]
        y = post_pad(y)
        y_l = np.count_nonzero(y)

        return x, x_l, y, y_l




if __name__ == '__main__':
    pass
