on paper by Lee et al (2016)

Fully Character-Level Neural Machine Translation without Explicit Segmentation, [annotated](https://drive.google.com/open?id=0ByV7wn2NzevOQ0JtTTRuR0pjUlE), [arXiv](https://arxiv.org/abs/1610.03017)

[Theano implementation by author Lee](https://github.com/nyu-dl/dl4mt-c2c?utm_campaign=Revue%20newsletter&utm_medium=Newsletter&utm_source=revue)

## Background
- Most MT research exclusively at word level
- NMT suffer from out-of-vocab words with languages with rich morphology
- Character-level better suited for multilingual translation than word-level because:
  - Does not suffer from out-of-vocab issues
  - Can model rare morphological variants of a word
  - No segmentation required
- Recent trend in MT is NMT with encoder, decoder and attention mechanism:
  - Encoder: Bidirectional RNN, concat of forward and backward hidden states
  - Attention: lets decoder attend more to differnet source symbols for each target symbol. There is a context vector $c_{t'}$ for each time step $t'$ as weighted sum of hidden states, ie the weights reflect relevance of inputs to the t'-th target token
  - Decoder: at time $t'$, computes hidden state $s_{t'}$ as a function of previous prediction, previous hidden and source context vector $c_{t'}$. Note how the context vector is specific for that output time step. Next, the prediction is produced by a parametric function (like beam search)
- Loss: model is trained to minimize the negative conditional log-likelihood of the probability of output target given previous target and input.
- Some other work based on character, but mostly subword-to-subword, or subword-to-character. Here they propose fully character-to-character

## Chararacter level challenges
- Sentences are much longer
- The decoder softmax operation is much faster over characters
- Attention mechanism with characters grows quadratically
- Encoder must encode long sequence of chars to good representation

## Model
- Aggressively uses convolutions + pooling to shorten input and capture local regularities
- **Encoder**: 
  - Char embedding size 128
  - 1D narrow convolution over padded sentence -> output length = input length
  - Various filter sizes from width 1 to 8 (up to char n-gram of 8)
  - Output of conv op is $Y \in \mathbb R^{N \times T_x}$, where $N$ is number of filter sizes, and $T_x$ is input length
  - Max pooling over time over $Y$, without mixing between widths in $N$, with stride $s$. So new $Y \to Y' \in \mathbb R^{N \times (T_x/s)}, where $s$ was chosen to be 5.
  - Highway network over pooling output to regulate information flow
  - Finally goes to BiGRU
- **Decoder**:
  - Attention and decoder like [NMT](summaries/neural_machine_translation.md) model, but predict characters as opposed to words.
  - Two-layer unidirection with 1024 GRU, beam search with width 20
- See Table 2 for full model parameters

## Experiment Setup
- Char2char model, includes only sentences with max 450 chars.
- Adam optimizer, $\alpha$ of 0.0001 and minibatch size 64
- Gradient clipping with threshold of 1
- Weights initialized from uniform [-0.01, 0.01]
- Multilingual char2char and bilingual char2char
- A few sub-word models as baseline
- Data scheduling to avoid overfitting to one language

## Observations
- Char2char always outperforms
- For some language bilingual char2char outperforms, in others the multilingual char2char outperforms
- BLEU metrics encourage reference-like translations, so additional evaluation by humans on adequacy and fluency
- Translation improvement by char2char mainly from fluency
- Two weeks to train char2char
- Char2char model not told any concept of word boundary, automatically learns
