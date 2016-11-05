acl paper [Implicit Discourse Relation Detection via a Deep Architecture with Gated Relevance Network](https://www.aclweb.org/anthology/P/P16/P16-1163.pdf)

## Problem
- Discourse relation recognition, easy for explicit, difficult for implicit.
- Traditionally use word pairs, such as "warm, cold", but data is sparse.

## Model
- Word2Vec embeddings, then Bidirecitonal LSTM to represent input over two separate text units X and Y
- Gated Relevance Network (GRN):
  - Get positional representation from BiLSTM output
  - Compute relevance score between every pair of x and y, with Bilinear Model and Single Layer Network.
  - Bilinear Model: let $(h_{xi}, h_{yi})$ be the vectorized representation of from the BiLSTM of from X and Y. Bilinear Model is function: $s(h_{xi}, h_{yi} = h_{xi}^TMh_{yj$, where $M \in \mathbb R^{d_h \times d_h}$ is the matrix coefficient to learn. Note the relationship between the two is linear. 
  - The Single Layer Network captures nonlinear interaction: standard single hidden neural net where output is nonlinear function over input plus bias, where input is concat of the pair. 
  - The two models are incorporated through the gate mechanism. The output of the GRN is :gate * linear + (1-gate)*non-linear. So the gate controls flow from linear and non-linear. 
  - GRN is similar to Neural Tensor Network (Socher et al 2013) but with added gate.
  - Output of GRN is an interaction score matrix
- Max pooling over GRN output which feeds into dense hidden layer (MLP), and finally connects to output.
- Train four binary classifiers to identify top level relations

## Observations
- LSTM alone has poor performance, loses too much local information
- Cosine distance, Bilinear or Single Layer alone do not perform very well
- Boost in LSTM when encoding to positional representation, boost with Bilinear and Single Layer relevance scores, and boost with extra gating on the relevance score. The mixture of scores performs best
- Model performs best in all categories
- Using the BiLSTM to encode rather than just words directly gets much better emphasis on relevance scores(Figure 3). Important word pairs for discourse are emphasised and irrelevant ones ignored.
