// FROM: http://karpathy.github.io/neuralnets/

var print = function(str) {
  str_new = document.getElementById('output').innerHTML + "<br>" + str;
  document.getElementById('output').innerHTML = str_new;
}

/* 
More realistic example, we have a single neuron which activates with sigmoid:
f(x,y,a,b,c) = \sigma(ax + by + c)

Sigmoid squashes values to between 0 and 1. 
The partial derivative with respect to a single input:
\frac{\partial \sigma(x)}{\partial x} = \sigma(x) (1 - \sigma(x))
*/

// --------------------------------------------------------------------------
// UNIT CLASS
// --------------------------------------------------------------------------
// every Unit corresponds to a wire in the diagrams
var Unit = function(value, grad) {
  // value computed in the forward pass
  this.value = value;
  // the derivative of circuit output w.r.t this unit, computed in backward pass
  this.grad = grad;
}

// --------------------------------------------------------------------------
// GATE CLASSES: FORWARD/BACKWARD DEFINITION
// --------------------------------------------------------------------------
// the backward functions compute only the local derivatives

// MULTIPLY GATE
var multiplyGate = function() {};
multiplyGate.prototype = {
  forward: function(u0, u1) {
    // From two input units, multiply their values and forward result in new parent Unit
    // store pointers to input Units u0 and u1 and output unit utop
    this.u0 = u0;
    this.u1 = u1;
    this.utop = new Unit(u0.value * u1.value, 0.0);
    return this.utop;
  },
  backward: function() {
    // take the gradient in output unit and chain it with the
    // local gradients, which we derived for multiply gate before
    // then write those gradients to those Units.
    this.u0.grad += this.u1.value * this.utop.grad;
    this.u1.grad += this.u0.value * this.utop.grad;
    // remember in multiplication the gradient wrt u0 is u1.value
  }
}

// ADD GATE
var addGate = function() {};
addGate.prototype = {
  forward: function(u0, u1) {
    this.u0 = u0;
    this.u1 = u1; // store pointers to input units
    this.utop = new Unit(u0.value + u1.value, 0.0);
    return this.utop;
  },
  backward: function() {
    // add gate. derivative wrt both inputs is 1
    this.u0.grad += 1 * this.utop.grad;
    this.u1.grad += 1 * this.utop.grad;
  }
}

// SIGMOID GATE
var sigmoidGate = function() {
  // helper function
  this.sig = function(x) {
    return 1 / (1 + Math.exp(-x));
  };
};
sigmoidGate.prototype = {
  forward: function(u0) {
    this.u0 = u0;
    this.utop = new Unit(this.sig(this.u0.value), 0.0);
    return this.utop;
  },
  backward: function() {
    var s = this.sig(this.u0.value);
    this.u0.grad += (s * (1 - s)) * this.utop.grad;
  }
}

// --------------------------------------------------------------------------
// MAIN
// --------------------------------------------------------------------------
// create input units
var a = new Unit(1.0, 0.0);
var b = new Unit(2.0, 0.0);
var c = new Unit(-3.0, 0.0);
var x = new Unit(-1.0, 0.0);
var y = new Unit(3.0, 0.0);

// create the gates
var mulg0 = new multiplyGate(); // to multiply ax
var mulg1 = new multiplyGate(); // to multiply by
var addg0 = new addGate(); // to add ax + by
var addg1 = new addGate(); // to add c to ax+by
var sg0 = new sigmoidGate(); // sigmoid result of previous

// Do the forward pass
var forwardNeuron = function() {
  ax = mulg0.forward(a, x); // a*x = -1
  by = mulg1.forward(b, y); // b*y = 6
  axpby = addg0.forward(ax, by); // a*x + b*y = 5
  axpbypc = addg1.forward(axpby, c); // a*x + b*y + c = 2
  s = sg0.forward(axpbypc); // sig(a*x + b*y + c) = 0.8808
};

// INITIAL FORWARD
print('initial values:');
print('a: ' + a.value);
print('b: ' + b.value);
print('c: ' + c.value);
print('x: ' + x.value);
print('y: ' + y.value);
forwardNeuron();
print('initial circuit output: ' + s.value); // prints 0.8808

// LOOP!
print_every = 10
for (i = 0; i < 200; i++) {
  // BACKWARD
  s.grad = 1.0;
  sg0.backward(); // writes gradient into axpbypc
  addg1.backward(); // writes gradients into axpby and c
  addg0.backward(); // writes gradients into ax and by
  mulg1.backward(); // writes gradients into b and y
  mulg0.backward(); // writes gradients into a and x

  // UPDATE
  var step_size = 0.01;
  a.value += step_size * a.grad; // a.grad is -0.105
  b.value += step_size * b.grad; // b.grad is 0.315
  c.value += step_size * c.grad; // c.grad is 0.105
  x.value += step_size * x.grad; // x.grad is 0.105
  y.value += step_size * y.grad; // y.grad is 0.210

  // FORWARD RESULT
  forwardNeuron();
  if (i % print_every == 0) {
    print('output at step ' + i + ' : ' + s.value);
  }
}

// FINAL
print('final values:');
print('a: ' + a.value);
print('b: ' + b.value);
print('c: ' + c.value);
print('x: ' + x.value);
print('y: ' + y.value);
print('final output : ' + s.value);

