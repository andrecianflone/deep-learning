Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks, [arXiv](https://arxiv.org/abs/1506.03099) Samy Bengio, Oriol Vinyals, Navdeep Jaitly, Noam Shazeer, 2015

TLDR; Scheduled sampling improves the quality of language generation by being more robust to mistakes. Use inverse sigmoid decay.

## Problem
One of the issues with training RNNs for prediction is that each $y$ being predicted is in part conditioned on previous *true* $y$, whereas at inference time $y$ is conditioned on a *generated* $y$. This can yield errors that accumulate throughout the prediction process. The authors propose a training method, "scheduled sampling", to sometimes condition on the true and sometimes on the generated.

Traditional inference is conditioned on the *most likely* previous prediction. Prediction error can compound through the entire prediction. One way to deal with this is to use beam search, which maintains several probable sequences in memory. Beam search produces $k$ "best" sequences, as opposed to searching through all possibilities of $Y$. 

## Scheduled Sampling
The authors propose a "curriculum learning" approach that forces the model to deal with mistakes. This is interesting since error correction is baked into the model.

While training the sampling mechanism randomly decides to use $y_{t-1}$ or $\hat{y}_{t-1}$. The true previous $y_{t-1}$ token is used with probability $\epsilon_i$. So if $\epsilon_i = 1$, we are using the same training as usual, and when $\epsilon_i = 0$, we are always training on predicted values. *Curriculum learning* strategy for training selects the true previous most of the time and slowly shifts to selecting a predicted previous. At the end of training, $\epsilon_i$ should favor sampling from the model. (See Figure 1).

The sampling variable $\epsilon$ decays according to a few schemes such as linear decay, exponential decay and inverse sigmoid decay (figure 2).

## Experiments
### Image Captioning
- Trained on MSCOCO, 75k for training and 5k for dev set.
- Each image has 5 possible captions, one is chosen at random.
- Image preprocessed by pretrained CNN
- Word generation done with LSTM(512), vocabular size is 8857
- Used inverse sigmoid decay

This approach led the team to first place for MSCOCO captioning challenge 2015.

### Constituency Parsing
Map a sentence onto a parse tree. Unlike image captioning, the task is much more deterministic, "uni-modal". Generally only one correct parse tree.

- One layer LSTM(512)
- Words as embeddings of size 512
- Attention mechanism
- Inverse sigmoid decay

### Speech Recognition

- Two layers of LSTM(250)
- Baseline trained 14 epochs, scheduled sampling only needed 9 epochs.
