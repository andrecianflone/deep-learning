on seminal Cho et al [Encoder-Decoder paper](https://arxiv.org/abs/1406.1078)

- [Sequence-to-sequence](https://www.tensorflow.org/versions/r0.11/tutorials/seq2seq/index.html) in Tensorflow
- Elaborate [seq2seq](https://github.com/farizrahman4u/seq2seq) library for Keras

## Model
- Two RNN that act as encoder and decoder pair
- The two networks are trained jointly to maximize the conditional probability of the output given the input
- Encoder: scan linearily input x, at each symbol update the hidden state of the RNN. At the end of the input, hidden state $c$ is a summary of the whole input sequence
- Decoder: Generative model. Predict next $y_t$ given the hidden state $h_t$. However, unlike Encoder, $y_t$ and $h_t$ are both conditioned on $y_{t-1}$ and $c$. 
- The Encoder-Decoder, once trained, can be used to either:
  - Generate output translation from input
  - Score a given pair of input and output sequences

## Details
- Uses new type of hidden unit, motivated by LSTM
- Reset gate: when reset gate is close to 0, the hidden state is forced to ignore the previous hidden state and reset with the current input only. This allows the hidden sate to **drop** any information that is found to be irrelevant later in the future, allowing more compact representation
- Update gate: controls how much information from the previous hidden state will carry over to the current hidden state. Helps remember long-term information like memory cell in LSTM

## Observations
- Gating is crucial. Without gating, just using tanh does not give meaningful results.
