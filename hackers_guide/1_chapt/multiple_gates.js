// FROM: http://karpathy.github.io/neuralnets/

var print = function(str) {
    str_new = document.getElementById('text').innerHTML + "<br>" + str;
  document.getElementById('text').innerHTML = str_new;
}

/* 
Like the base case, we expand the gradient calculations to multiple gates where each calculates local derivatives, unaware of the complexity of the whole.
*/

// ----------------------------------------------------------
// FUNCTION DEFINITION
// ----------------------------------------------------------
// We want to model the following expression:
// f(x,y,z) = (x + y) z
var forwardMultiplyGate = function(a, b) { 
  return a * b;
};
var forwardAddGate = function(a, b) { 
  return a + b;
};
var forwardCircuit = function(x,y,z) { 
  var q = forwardAddGate(x, y);
  var f = forwardMultiplyGate(q, z);
  return f;
};

var x = -2, y = 5, z = -4;
var f = forwardCircuit(x, y, z); // output is -12
print(f);

// ----------------------------------------------------------
// GRADIENTS VIA CHAIN RULE (SIMPLE BACKPROPAGATION)
// ----------------------------------------------------------
/*
To calculate the derivatives, we would start with the partial derivatives of function f (multiply gate) with respect to q and z, we would then calculate the partial derivatives of function q (add gate) with respect to x and y. 
We then combine the two via the chain rule to get the gradient with respect to x, y and z. 
*/
// initial conditions
var x = -2, y = 5, z = -4;
var q = forwardAddGate(x, y); // q is 3
var f = forwardMultiplyGate(q, z); // output is -12

// gradient of the MULTIPLY gate with respect to its inputs
// wrt is short for "with respect to"
var derivative_f_wrt_z = q; // 3
var derivative_f_wrt_q = z; // -4

// derivative of the ADD gate with respect to its inputs
var derivative_q_wrt_x = 1.0;
var derivative_q_wrt_y = 1.0;

// chain rule
var derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q; // -4
var derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q; // -4
/* Note although derivatives of q with respect to x,y are 1, gradient of f with respect to q is -4! Since q is made up of x,y and we want q to decrease, therefore x,y must decrease to respect q's gradient! This is why derivative of f wrt to x,y (via chain rule) is -4. */ 

// final gradient, from above: [-4, -4, 3]
var gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

// let the inputs respond to the force/tug:
var step_size = 0.01;
x = x + step_size * derivative_f_wrt_x; // -2.04
y = y + step_size * derivative_f_wrt_y; // 4.96
z = z + step_size * derivative_f_wrt_z; // -3.97

// Our circuit now better give higher output:
var q = forwardAddGate(x, y); // q becomes 2.92
var f = forwardMultiplyGate(q, z); // output is -11.59, up from -12! Nice!

// ----------------------------------------------------------
// GRADIENT CHECKING
// ----------------------------------------------------------
// initial conditions
var x = -2, y = 5, z = -4;

// numerical gradient check
var h = 0.0001;
var x_derivative = (forwardCircuit(x+h,y,z) - forwardCircuit(x,y,z)) / h; // -4
var y_derivative = (forwardCircuit(x,y+h,z) - forwardCircuit(x,y,z)) / h; // -4
var z_derivative = (forwardCircuit(x,y,z+h) - forwardCircuit(x,y,z)) / h; // 3
// We get [-4, -4, 3], same as with backpropagation




