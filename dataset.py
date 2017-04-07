import numpy as np



class Dataset(object):
    
    def __init__(self, source, target, vocab, batch_size=64, max_seq_len=30):
        self.vocab_map = self.build_vocab_mapping(vocab)

        self.vocab_size = len(self.vocab_map)

        self.source_data = self.prepare_data(source)
        self.target_data = self.prepare_data(target)

        self.train_indices, self.val_indices, self.test_indices = self.make_splits(len(self.source_data))

        self.batch_index = 0
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def make_splits(self, N):
        indices = np.arange(N)
        train_test_n = N / 8

        train = indices[:N - (train_test_n * 2)]
        val = indices[len(train): N - train_test_n]
        test = indices[len(train) + len(val):]

        return train, val, test


    def build_vocab_mapping(self, vocabfile):
        out = {w.split()[0].strip(): i+1 for (i, w) in enumerate(open(vocabfile))}
        out['<pad>'] = 0
        return out
        
    def prepare_data(self, corpusfile):
        dataset = []
        for l in open(corpusfile):
            dataset.append([self.vocab_map.get(w, self.vocab_map['<unk>']) for w in l.split()])
        return dataset

    def batch_iter(self, train=True):
        indices = self.train_indices if train else self.test_indices

        while self.has_next_batch(indices):
            yield self.get_batch(indices)
            self.batch_index += self.batch_size

        self.batch_index = 0


    def has_next_batch(self, indices):
        return self.batch_index + self.batch_size < len(indices)


    def get_batch(self, indices):
        def post_pad(x, pad=0):
            new =  [pad] * self.max_seq_len
            new[:len(x)] = x
            return new[:self.max_seq_len]

        x_batch = self.source_data[self.batch_index : self.batch_index + self.batch_size]
        x_batch = [post_pad(x) for x in x_batch]
        x_lens = np.count_nonzero(np.array(x_batch), axis=1).tolist()

        y_batch = self.target_data[self.batch_index : self.batch_index + self.batch_size]
        y_batch = [post_pad(y) for y in y_batch]
        y_lens = np.count_nonzero(np.array(y_batch), axis=1).tolist()
        
        return x_batch, x_lens, y_batch, y_lens

if __name__ == '__main__':
    d = Dataset('train.en', 'train.vi', 'vocab.en', 'vocab.vi')

    for x, xl, y, yl in d.batch_iter():
        continue
