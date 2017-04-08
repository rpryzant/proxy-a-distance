
## Proxy A-Distance

This is an implementation of an algorithm discussed in [Ganin et. al (2015)](https://arxiv.org/abs/1505.07818), [Glorot et. al (2011)](http://www.icml-2011.org/papers/342_icmlpaper.pdf), and [Ben-David et. al (2007)](https://papers.nips.cc/paper/2983-analysis-of-representations-for-domain-adaptation). It has been adapted for use with machine translation datasets, and released to the public under the MIT license. 

This algorithm computes the Proxy A-Distance (PAD) between two domain distributions. PAD is a measure of similarity between datasets from different domains (e.g. newspapers and talk shows). The lower the PAD, the closer two datasets are. 



## Requirements

* numpy: `pip install numpy`
* sklearn: `pip install sklearn`

## Usage

```
python main.py [corpusfile 1] [corpusfile 2] [vocab file]
```

* `corpusfile 1` is a text file with one sentence per line.
* `corpusfile 2` is another text file with one sentence per line.
* `vocab` is a text file with one token per line. These tokens represent a shared vocabulary for the above corpusfiles.
