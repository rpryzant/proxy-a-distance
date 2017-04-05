import numpy as np



class Dataset(object):
    
    def __init__(self, source, target, source_vocab, target_vocab, batch_size=64, max_seq_len=30):
        self.source_vocab_mapping = self.build_vocab_mapping(source_vocab)
        self.target_vocab_mapping = self.build_vocab_mapping(target_vocab)

        self.source_vocab_size = len(self.source_vocab_mapping)
        self.target_vocab_size = len(self.target_vocab_mapping)

        self.source_data = self.prepare_data(source, self.source_vocab_mapping)
        self.target_data = self.prepare_data(target, self.target_vocab_mapping)

        self.train_indices, self.test_indices = self.make_splits(len(self.source_data))

        self.batch_index = 0
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

    def make_splits(self, N):
        indices = np.arange(N)
        train = indices[:-N/8]
        test = indices[-N/8:]
        return train, test

    def build_vocab_mapping(self, vocabfile):
        out = {w.strip(): i+1 for (i, w) in enumerate(open(vocabfile))}
        out['<pad>'] = 0
        return out
        
    def prepare_data(self, corpusfile, vocab_map):
        dataset = []
        for l in open(corpusfile):
            dataset.append([vocab_map.get(w, vocab_map['<unk>']) for w in l.split()])
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
