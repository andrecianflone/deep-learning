## Overview optimization algorithms
[blog post](http://sebastianruder.com/optimizing-gradient-descent/index.html#gradientdescentvariants)

### Gradient descent
#### Batch gradient descent
- Computes the gradient of the cost on entire training set, in one go
- Minus
  - Slow as single update requires measuring loss on all data
  - Converts to global minimum only for convex error surfaces

#### Stochastic gradient descent (SGD)
- Plus
  * Parameter update after each training sample, faster than batch
  * Frequent updates cause objective function to fluctuate
  * Can easily overshoot minimum, so better to decay learning rate
- Should shuffle data at each epoch

#### Mini-batch gradient descent
- Best of both, size should be 50 ~ 256
- Plus
  * Reduces variance of parameter updates
  * Efficient computation of matrices
- Minus
  * Same issues with learning rate, can address problem with schedules
  * Same learning rate applies to all parameters, problem
  * Easy to get trapped in local minima and saddle points

### Optimization (adaptive methods)
#### Momemtum
- Helps SGD accelerate in relevant direction and dampens oscillations
- Adds fraction of previous step vector to current step vector

#### Nesterov accelerated gradient (NAG)
- Lookahead descent, slows down before hill slopes up
- Calculates momentum as well as approximation of next parameter value

#### Adagrad
- Adapts learning rate to the parameters
  * Large updates for infrequent parameters
  * Small updates for frequent parameters
  * Different rate for every theta at every time step
- Plus
  * Good for sparse data
  * No longer need to tune learning rate
- Minus
  * Because of accumulated sum in denom, learning rate shrinks and vanishes

#### Adadelta
- Extends Adagrad, fixes decreasing learning rate
- Restricts window of accumulated past gradients

#### Adaptive Moment Estimation (Adam)
- Also computes adaptive learning for each parameter
- Like Adadelta, stores decaying average of past squared gradients
- Also stores decayin average of past gradients
- Basically stores first two moments

### Tips
- Shuffle before each epoch
- For progressively harder problems, use curriculum learning
- Batch normalization so can use higher learning rates
- Early stopping: stop if error no longer improves
- Add gaussian noise to gradients
