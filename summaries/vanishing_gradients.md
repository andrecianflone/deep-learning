[Summary of this section](http://neuralnetworksanddeeplearning.com/chap5.html#the_vanishing_gradient_problem)

- Train on MNIST dataset
- Base network: [784, 30, 10], 784 input, 30 node hidden, 10 class classifier
  * Accuracy: 0.9648
- Expand network: [784, 30, 30, 10]
  * Improvement in Accuracy: 0.9690
- Expand network: [784, 30, 30, 30, 10]
  * Accuracy drops: 0.9657

# Vanishing gradients
- For a two hidden-layer net, second layer neurons learn much faster than neurons in the first layer
