Sutskever et al paper [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215), [annotated](https://drive.google.com/file/d/0ByV7wn2NzevOQ1l5aUF4RWYtenc/view?usp=sharing)

- [Sequence-to-sequence](https://www.tensorflow.org/versions/r0.11/tutorials/seq2seq/index.html) in Tensorflow
- Elaborate [seq2seq](https://github.com/farizrahman4u/seq2seq) library for Keras

## Background
- Some problems can be seen as sequence to sequence problems, mapping an input sequence to an output sequence. Such as translation, question and answering, etc. 
- One challenge for DNN is dimensionlity of input/output which must be fixed. This can be overcome with LSTMs

## Model
- Model maps input ABC to output WXYZ. Does not use the RNN for scoring as Cho et al, but to produce the translation
- After training, produce translation with left-to-right beam-search decoder
- Ensemble of 5 deep LSTMs with beam of size 2
- Reverse the order of the input, leave order of output
- Use two LSTMs, much like Encoder-Decoder
- 4 layer LSTMs. Deep LSTMs significantly outperformed shallow LSTMs
- 1000 cells at each layer
- 1000 dimensional embeddings

## Experiment
- Used the LSTM to rescore publicly available 1000-best lists of the SMT baseline and obtained close to SOTA results.
- Much better results when inversing input. Test perplexity drops from 5.8 to 4.7 and test BLEU increases from 25.9 to 30.6!
- Used 160k most frequent words for source language and 80k most frequent for the target language
- Predict a small number $B$ most likely partial hypothesis with beam search. If a hypothesis is appended with "EOS", the hypothesis is added to set of complete hypothesis. Beam search continues until all partial hypothesis are complete.
- Most sentences are short, some are very lond, which can waste computation in minibatch. Therefore, made sure sentences in minibatch are roughly same length, yielding 2x speedup.
- Parallelized on 8 GPUs: one LSTM layer per GPU (for 4 layers), and other 4 GPUs for softmax calculation. Results in 3.7x speedup.
- Ensemble of 5 LSTMs with beam of size 2 is cheaper than a single LSTM with a beam of size 12
- Surprisingly, performs well on long sentences

