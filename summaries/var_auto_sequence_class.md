about paper [Semi-supervised Variational Autoencoders for Sequence Classification](https://arxiv.org/abs/1603.02514), [annotated](https://drive.google.com/file/d/0ByV7wn2NzevOTXEzLWlNQy1od0k/view?usp=sharing).

## Problem
- SemiVAE work well in image classification tasks, but fail for text classification if using vanilla LSTM as conditional generative model.
- We have more and more data, but very little labels accompanying the data
- We want unsupervised to extract useful features which we can then use in supervised tasks
- RNNs are good for sequence-to-sequence, but not good for high level features like topic, style, and sentiment. Variational Recurrent Autoencoders have been used for this.

## Background
- Conditional variational autoencoders can generate samples according to certain attributions of given labels.

## The Model
- Novel semi-supervised deep generative model for text classification, the model can generate sentences conditioned on labels
- A RNN encodes input text x with the conditional input y (the label). The generative network then decodes the latent variable z, where $z \sim p(z|x,y)$.
- Conditional LSTM network proposed as conditional generative model. Same traditional LSTM equations except one has extra term about $y$.

