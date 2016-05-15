// FROM: http://karpathy.github.io/neuralnets/

var print = function(str) {
  str_new = document.getElementById('output').innerHTML + "<br>" + str;
  document.getElementById('output').innerHTML = str_new;
}

// --------------------------------------------------------------------
// TRAINING DATA
// --------------------------------------------------------------------
var data = []; var labels = [];
data.push([1.2, 0.7]); labels.push(1);
data.push([-0.3, -0.5]); labels.push(-1);
data.push([3.0, 0.1]); labels.push(1);
data.push([-0.1, -1.0]); labels.push(-1);
data.push([-1.0, 1.1]); labels.push(-1);
data.push([2.1, -3]); labels.push(1);

/*
Code below is simple neural network, 3 hidden neurons and one output layer. The code is not modular like the other js code.

Activation function is ReLU, which is simply max(0, x).
*/
// --------------------------------------------------------------------
// RANDOMLY INITIALIZE NETWORK PARAMETERS
// --------------------------------------------------------------------
// Neuron 1
var a1 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var b1 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var c1 = Math.random() - 0.5; // a random number between -0.5 and 0.5

// Neuron 2
var a2 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var b2 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var c2 = Math.random() - 0.5; // a random number between -0.5 and 0.5

// Neuron 3
var a3 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var b3 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var c3 = Math.random() - 0.5; // a random number between -0.5 and 0.5

// Output layer
var a4 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var b4 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var c4 = Math.random() - 0.5; // a random number between -0.5 and 0.5
var d4 = Math.random() - 0.5; // a random number between -0.5 and 0.5

// --------------------------------------------------------------------
// FORWARD/BACKWARD/UPDATE
// --------------------------------------------------------------------
for(var iter = 0; iter < 400; iter++) {
  // pick a random data point
  var i = Math.floor(Math.random() * data.length);
  var x = data[i][0];
  var y = data[i][1];
  var label = labels[i];

  // compute forward pass
  var n1 = Math.max(0, a1*x + b1*y + c1); // activation of 1st hidden neuron
  var n2 = Math.max(0, a2*x + b2*y + c2); // 2nd neuron
  var n3 = Math.max(0, a3*x + b3*y + c3); // 3rd neuron
  var score = a4*n1 + b4*n2 + c4*n3 + d4; // the score
  if (iter % 25 === 0) print(score)

  // compute the pull on top
  var pull = 0.0;
  if(label === 1 && score < 1) pull = 1; // we want higher output! Pull up.
  if(label === -1 && score > -1) pull = -1; // we want lower output! Pull down.

  // now compute backward pass to all parameters of the model

  // backprop through the last "score" neuron
  // First backprop is simple deriv of a*b type, ie da = x * top_gradient
  var dscore = pull;
  var da4 = n1 * dscore;
  var dn1 = a4 * dscore;
  var db4 = n2 * dscore;
  var dn2 = b4 * dscore;
  var dc4 = n3 * dscore;
  var dn3 = c4 * dscore;
  var dd4 = 1.0 * dscore; // phew

  // backprop the ReLU non-linearities, in place
  // i.e. just set gradients to zero if the neurons did not "fire"
  var dn3 = n3 === 0 ? 0 : dn3;
  var dn2 = n2 === 0 ? 0 : dn2;
  var dn1 = n1 === 0 ? 0 : dn1;

  // backprop to parameters of neuron 1
  var da1 = x * dn1;
  var db1 = y * dn1;
  var dc1 = 1.0 * dn1;

  // backprop to parameters of neuron 2
  var da2 = x * dn2;
  var db2 = y * dn2;
  var dc2 = 1.0 * dn2;

  // backprop to parameters of neuron 3
  var da3 = x * dn3;
  var db3 = y * dn3;
  var dc3 = 1.0 * dn3;

  // phew! End of backprop!
  // note we could have also backpropped into x,y
  // but we do not need these gradients. We only use the gradients
  // on our parameters in the parameter update, and we discard x,y

  // add the pulls from the regularization, tugging all multiplicative
  // parameters (i.e. not the biases) downward, proportional to their value
  da1 += -a1; da2 += -a2; da3 += -a3;
  db1 += -b1; db2 += -b2; db3 += -b3;
  da4 += -a4; db4 += -b4; dc4 += -c4;

  // finally, do the parameter update
  var step_size = 0.01;
  a1 += step_size * da1;
  b1 += step_size * db1;
  c1 += step_size * dc1;
  a2 += step_size * da2;
  b2 += step_size * db2;
  c2 += step_size * dc2;
  a3 += step_size * da3;
  b3 += step_size * db3;
  c3 += step_size * dc3;
  a4 += step_size * da4;
  b4 += step_size * db4;
  c4 += step_size * dc4;
  d4 += step_size * dd4;
  // wow this is tedious, please use for loops in prod.
  // we're done!
}
