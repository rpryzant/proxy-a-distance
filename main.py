"""
python main.py data/europarl_opensubtitles_processed/test.combined.bpe.en data/europarl_opensubtitles_processed/test.combined.bpe.fr data/europarl_opensubtitles_processed/bpe.vocab

"""


import argparse # option parsing
from dataset import Dataset



def process_command_line():
  """
  Return a 1-tuple: (args list).
  `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
  """

  parser = argparse.ArgumentParser(description='usage') # add description
  # positional arguments
  parser.add_argument('d1', metavar='domain-1', type=str, help='domain 1 corpus')
  parser.add_argument('d2', metavar='domain-2', type=str, help='domain 2 corpus')
  parser.add_argument('v', metavar='vocab', type=str, help='shared bpe vocab')

  args = parser.parse_args()
  return args

def main(domain1, domain2, vocab):
    d = Dataset(domain1, domain2, vocab)




if __name__ == '__main__':
    args = process_command_line()

    main(args.d1, args.d2, args.v)
