on seminal Cho et al [Encoder-Decoder paper](https://arxiv.org/abs/1406.1078)

- [Sequence-to-sequence](https://www.tensorflow.org/versions/r0.11/tutorials/seq2seq/index.html) in Tensorflow
- Elaborate [seq2seq](https://github.com/farizrahman4u/seq2seq) library for Keras

## Model
- Two RNN that act as encoder and decoder pair
- The two networks are trained jointly to maximize the conditional probability of the output given the input
- Encoder: scan linearily input x, at each symbol update the hidden state of the RNN. At the end of the input, hidden state $c$ is a summary of the whole input sequence
- Decoder: Generative model. Predict next $y_t$ given the hidden state $h_t$. However, unlike Encoder, $y_t$ and $h_t$ are both conditioned on $y_{t-1}$ and $c$. 
- The Encoder-Decoder, once trained, can be used to either:
  - Generate output translation given input. or,
  - Score a given pair of input and output sequences produced by other algorithm

## Details
- Uses new type of hidden unit, motivated by LSTM
- Reset gate: when reset gate is close to 0, the hidden state is forced to ignore the previous hidden state and reset with the current input only. This allows the hidden sate to **drop** any information that is found to be irrelevant later in the future, allowing more compact representation
- Update gate: controls how much information from the previous hidden state will carry over to the current hidden state. Helps remember long-term information like memory cell in LSTM
- Encoder-Decoder ignores normalized frequencies of phrase pairs in original corpora

## Experiment
- Uses WMT dataset. Data from Europarl, UN, crawled data. For language model only, uses crawled data.
- For RNN Encoder-Decoder, limits source and target vocabulary to most frequent 15k words, about 93% coverage
- RNN E-D has 1000 hidden units in both encoder and decoder.
- Uses rank-100 matrices, equivalent to learning an embedding of dimension 100 for each word.
- Uses tanh as activation for new hiddent state

## Observations
- Gating is crucial. Without gating, just using tanh does not give meaningful results.
- The model is focused towards learning linguistic regularities: distinguishing between plausible and implausible translations or manifold (regions of probability concentration) or plausible translations. 
