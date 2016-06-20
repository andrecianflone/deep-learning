# Generic ML datasets
- **Linear data** can be solved with linear classifiers sur as logistic regression, svm, and so on.
- **Moon data** has two clusters. Linear classifier cannot fully seperate the two classes
- **Saturn data** has a core cluster and ring cluster, requires non-linear classifier

# Hidden layer, number of nodes
- Simple neural net with one hidden layer and two nodes will dramatically vary in accuracy (0.88~0.96) due to random initialization. Hidden layer with 3 nodes give consistent results around 0.97 accuracy.
- Sensitive to weight initialization: use one of two following
  * Truncated normals for weights, 0.1 for biases and ReLU
  * Xavier initialization, 0 biases, tanh
