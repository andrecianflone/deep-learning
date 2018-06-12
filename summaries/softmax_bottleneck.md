
**Breaking the Softmax Bottleneck: A High-Rank RNN Language Model**, Yang et al, ICLR 2018. [openreview](https://openreview.net/forum?id=HkwZSG-CZ), [arXiv](https://arxiv.org/abs/1711.03953)

Language modeling consists of learning the joint language distribution factorized as a product of word probabilities conditioned over context.

Given a language model output matrix A over time, where each row is the vocabulary distribution given context. A word logit is produced by the inner product of h_c (rnn hidden state) and w_x (embedding vector), both of dimension d.

The authors hypothesize A  must be high rank to express complex language. The rank of A can be as high as M, the vocabulary size. The single softmax is not expressive enough if d is too small, and thus it is learning a low-rank approximation of A.

A more expressive model would be Ngram or simply increasing d, however these approaches lead to significant increases in parameters and hurts generality.

They propose a mixture of K softmax distributions (MoS). Each softmax distribution is weighted by pi_c, which is itself learned by the context. They empirically measure the MoS matrix A compared to single sofmax A and show that with a mixture of 15 softmax trained on PTB, A's rank is as high as M! They get rank 9981, while the single softmax is 400 (Table 6). They also achieve SOTA on PTB and WikiText-2

