on paper by Bahdanau, Cho, Bengio, ICLR 2015

Neural Machine Translation by Jointly Learning to Align and Translate, [annotated](https://drive.google.com/file/d/0ByV7wn2NzevOS3FmWHVNazhnczA/view?usp=sharing), [arXiv](https://arxiv.org/abs/1409.0473)

See TensorFlow library [seq2seq](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/seq2seq.py) for usage

## Background
- Unlike traditional phrase-based translation system which consist of many small sub-components that are tuned separately, NMT builds and trains single large neural net to encode/decode translation
- Issue with NMT encoder/decoder is compressing all source into single fixed-length vector

## Model
- Extends Encoder-Decoder by learning to align and translate
- Does not encode whole input to single fixed-vector, but encodes into a sequence of vectors. Decoder chooses a subset of these vectors adaptively while decoding the translation
- Traditional encoder-decoder, for each $y_t$, the decoder predicts the conditional probability of $y_t$ given all previous words and context. With RNN, this is estimated as $g(y_{t-1}, s_t, c)$ where: $g$ is a non-linear, $s_t$ is hidden state and $c$ is context vector from encoder
- New decoder is conditioned on distinct context vector $c_i$ for each target word $y_i$: $p(y_i|y_1,...,y_{i-1},x)=g(y_{i-1},s_i,c_i)
- Context $c_i$ is weighted sum of annotations $h_i$ from encoder (Eq. 5). The weight is a softmax over an alignment model. This effectively acts like attention mechanism, tells decoder what to pay attention to and relieves encoder burden of encoding all into fixed-length vector
- Decoder has 1000 hidden units and single maxout hidden layer to compute conditional probability of each target word
- Once trained, use beam search to find translation that maximizes the conditional probability
- Encoder is BiRNN, 1000 hidden units

## Experiments
- Trained on regular Encoder-Decoder (Cho et al. 2014), called RNNencdec and new proposed model named RNNsearch
- Trained on sentences of max 30 words (RNNencdec-30, RNNsearch-30), and max 50 words (RNNencdec-50, RNNsearch-50)
- In both cases, RNNsearch outperforms.

## Remarks
- Sequence of vectors and adaptive decoding frees model from having to squash all information of a source sentence into a fixed-length vector. Copes better with long sentences
- Alignment learning significantly improves basic encoder-decoder
- Astoundingly, RNNsearch-50 massively outperforms other models as sentences become very long, with no deterioration in BLEU score with sentences approaching length 60 where others go towards BLEU score 0.
- The alignment matrix generally shows monotonic correlations. However, number of non-trivial non-monotonic alignments among adjectives and nouns since order is different between English and French. Model correctly translated [European Economic Area] to [zone economique europeenne]
