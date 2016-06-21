[Summary of this section](http://neuralnetworksanddeeplearning.com/chap5.html#the_vanishing_gradient_problem)

## Exposing the problem
- Train on MNIST dataset
- Base network: [784, 30, 10], 784 input, 30 node hidden, 10 class classifier
  * Accuracy: 0.9648
- Expand network: [784, 30, 30, 10]
  * Improvement in Accuracy: 0.9690
- Expand network: [784, 30, 30, 30, 10]
  * Accuracy drops: 0.9657

In theory deep networks work better than shallow ones due to abstraction features as the network gets deeper. Deep networks could solve a problem with dramatically less parameters than shallow networks. But sometimes deep networks don't perform well. Sometimes, when later layers in a deep network work well, early layers get stuck, possibly learning nothing. The opposite may be true, with earlier layers learning well and later ones being stuck.

## Vanishing gradients (and exploding)
- For a two hidden-layer net, second layer neurons learn much faster than neurons in the first layer
- For each layer, let vector gl represent a vector of gradients for l-th layer where each entry determine how quickly the  hidden layer learns
- ||gl||, the length of the vector, determines the speed of learning of the l-th layer
- For two hidden-layer network, g1 = 0.07 and g2 = 0.31. The second layer learns faster
- For 3 hidden-layer netowrk, lengths are 0.012, 0.06, 0.283. Earlier layers are slower
- As we go deeper, gradients in earlier layers get much smaller, i.e. vanishing gradient
- Instead of vanishing, early layer gradients may explode

## Vanishing gradient cause (and exploding)
Simple structure with backprop
![network derivative](http://neuralnetworksanddeeplearning.com/images/tikz38.png)
Where $\sigma (z_j)$ is the sigmoid, and $zj = w_j a_{j-1} + b_j$ is the weighted input to the activation function in the next neuron. (See proof starting at formula 114). The expression is the partial derivative of the cost with respect to the first bias, b1. Besides the last term, it follows the pattern of weight x sigmoid derivative.

Looking at the sigmoid derivative plot:
![sigmoid deriv](http://www.billharlan.com/papers/logistic/img39.png)
- Notice it peaks at 1/4. 
- Weights are initialized with standard gaussian, mean 0 and standard deviation of 1
- Weights are less than 1
- Terms $|w_j \sigma^' (z_j)|$ less than 1/4
- The product of all these terms results in tiny gradient
- If the weights are initialized above 1, they grow exponentially as we move back through the layers, causing them to explode

In summary, choice of activation function, weight initilization, optimization algorithm and network architecture can cause unstable gradients.

