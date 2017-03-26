[Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080), Vinyals et al for NIPS 2016

## Problem
The problem is to learn something with few examples is a huge challenge for DNNs, when it can be trivial for humans. A child can generalize the concept of "giraffe" from just a single example. For machines, it's hard. This is a particular challenge for many NLP datasets [my opinion].

This motivates the "one-shot" and "few-shot" learning paradigm.

## Model
The authors present a model that tries to match an input \hat{x} with an example from a support set of image-label pairs S={(xi, yi)} of k examples. The closest example in the support set is thus the likeliest label yi. Then the classifier Cs(\hat{x}) defines a probability distribution over outputs \hat{y}.

The output \hat{y} is defined a the sum of a(\hat{x}, xi)yi, where a() is an attention function.

## Attention
- A proposed attention mechanism is the use of softmax over cosine distance c(f(\hat{x}), g(xi)).
- The functions f() and g() that embed the two inputs are neural networks such as deep ConvNets.
- They propose to modify the embedding function g to include the whole support S instead of only xi, this way the net can change an embedding if it deems it too close to another element in the set. g(xi) becomes g(xi,S).

