"""
python main.py test_data/dev.europarl.bpe.en test_data/dev.europarl.bpe.fr test_data/dev.opensubtitles.bpe.en test_data/dev.opensubtitles.bpe.fr test_data/bpe.vocab

"""


import argparse # option parsing
from dataset import Dataset
from models import NN, SVM
import random
import numpy as np

def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """

  parser = argparse.ArgumentParser(description='usage') # add description
  # positional arguments
  parser.add_argument('d1s', metavar='domain1-source', type=str, help='domain 1 source')
  parser.add_argument('d1t', metavar='domain1-target', type=str, help='domain 1 target')

  parser.add_argument('d2s', metavar='domain2-source', type=str, help='domain 2 source')
  parser.add_argument('d2t', metavar='domain2-target', type=str, help='domain 2 target')

  parser.add_argument('v', metavar='vocab', type=str, help='shared bpe vocab')

  # optional arguments
  parser.add_argument('-b', '--batch-size', dest='b', type=int, default=32, help='batch_size')

  args = parser.parse_args()
  return args



def main(domain1_source, domain1_target, domain2_source, domain2_target, vocab, batch_size):
    data_iterator = Dataset(domain1_source, domain1_target, domain2_source, domain2_target, vocab, batch_size=batch_size)
#    model = NN(batch_size, data_iterator.get_vocab_size())
    model = SVM(batch_size, data_iterator.get_vocab_size())

    model.fit(data_iterator)
    print model.test(data_iterator)
    # print type(x)
    # print x
    # print list(x)

    # data_iterator.batch_size = data_iterator.train_n
    # print data_iterator.batch_size
    # quit()
    # while True:
    #   for batch in data_iterator.mixed_batch_iter():
    #       b = batch
    #       model.train_on_batch(*batch)
    #       print [np.argmax(x) for x in batch[0]]
    #       print model.predict(*batch[1:])

#          l, preds = model.train_on_batch(*b)






if __name__ == '__main__':
    args = process_command_line()

    main(args.d1s, args.d1t, args.d2s, args.d2t, args.v, args.b)
