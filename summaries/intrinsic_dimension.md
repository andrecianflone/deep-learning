**Measuring the Intrinsic Dimension of Objective Landscapes**, Li et al, ICLR 2018. [openreview](), [arXiv](https://arxiv.org/abs/1804.08838)

tl;dr: Intrinsic Dimension is the minimal parameter subspace (projected to the total parameters) to achieve a certain performance. It is a measure of model-problem complexity.

First, let's describe the normal training of a neural network until convergence as the "direct method".
Consider an alternative to the direct method.
Given a set of model parameters, train a lower dimensional set of parameters which is then projected and added to the fixed larger set. The smaller set size is the "subspace". The size of the subspace controls the degree of freedom of the model. Train several subspace size, each time training until convergence. As you increase the subspace size, at one point accuracy jumps and you achieve 90% performance of the direct method (and NOT 90% accuracy on the task). The subspace size that achieves 90% is called the intrinsic dimension.

What's interesting is that when you increase the original model capacity, the subspace parameters are projected to a larger space, and yet you barely need to increase the subspace size to solve the same problem. For example, MNIST is always a subspace of around 750 on an MLP model. But if you change to a CNN, the subspace for MNIST is much smaller, around 250, showing the CNN is a superior model on this dataset.

Also, harder problems need larger subspace, like Pong from pixels vs MNIST. If we view solving MDPs and supervised learning as function approximation, then we can compare different problems with the intrinsic dimension metric. Apparently, Pong from pixels is equivalent to CIFAR-10!
