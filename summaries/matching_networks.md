[Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080), Vinyals et al, all from DeepMind, for NIPS 2016

## Problem
The problem is to learn something with few examples is a huge challenge for DNNs, when it can be trivial for humans. A child can generalize the concept of "giraffe" from just a single example. For machines, it's hard. This is a particular challenge for many NLP datasets [my opinion].

This motivates the "one-shot" and "few-shot" learning paradigm.

## Model
The authors present a model that tries to match an input \hat{x} with an example from a support set of image-label pairs S={(xi, yi)} of k examples.

The output \hat{y} is defined a the sum of a(\hat{x}, xi)yi, where a() is an attention function.

## Attention
- The proposed attention mechanism a() is to use softmax over cosine distance c(f(\hat{x}), g(xi)). The softmax normalized the cosine of \hat{x} and x by dividing by the sum of cosine over all x_j for j=1 to k. So a() is a weight for that class i which is multiplied by y_i. Therefore \hat{y} is a blend of the classes.
- For example, imagine we have 3 classes and a() has calculated a probability for each of 3 classes in yi: 0.3, 0.5 and 0.2. \hat{y} as a weighted sum would be equal to 0.2[1,0,0] + 0.5[0,1,0] + 0.3[0,0,1] = [0.2,0.5,0.3]. This should be more explicit in the paper.
- The functions f() and g() that embed the two inputs are neural networks such as deep ConvNets.
- They propose to modify the embedding function g to include the whole support S instead of only xi, this way the net can change an embedding if it deems it too close to another element in the set. g(xi) becomes g(xi,S).

## Results
The approach shows SOTA results on Omniglot, ImageNet and PTB. Note these are smaller datasets selected for the task. The PTB "mini" dataset is proposed by this paper. 

## Summary
In summary, the model predicts a class indirectly by mapping input samples to samples from an example set by taking their labels. The task for the model is thus to learn how to best represent samples (via a neural net) to compute distance metrics as effectively as possible. Once it becomes good at this, it can match samples with classes it has never even seen in training!
