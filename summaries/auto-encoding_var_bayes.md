[original paper](https://arxiv.org/abs/1312.6114) about var-bayes autoencoders by Kingma and Welling, 2013. Check out [implementation in Keras](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py) for example code

Also:
- [Presentation](https://home.zhaw.ch/~dueo/bbs/files/vae.pdf) about the paper to help understand.
- [Accompanying Python notebook](https://github.com/oduerr/dl_tutorial/tree/master/tensorflow/vae)

## Background
- How can we perform efficient approximate inference and learning with directed probabilistic models whose continuous latent variables and/or parameters have intractable posterior distributions?
- Variational Bayes optimizes an approximation of the intractable posterior
- See [lower bounds for estimation](https://www.stat.washington.edu/jaw/COURSES/580s/581/LECTNOTES/ch3-rev1.pdf)

## Problem
- Assume $X = {x^i }_{i=1}^N$. The process consists of getting $z^i$, generated from a prior $p_theta * (z)$ and value $x^i$ generated from conditional $p_theta * (x|z)$. 
- Algorithm must work in case of intractability and large dataset

