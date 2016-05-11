// FROM: http://karpathy.github.io/neuralnets/

var print = function(str) {
    str_new = document.getElementById('text').innerHTML + "<br>" + str;
  document.getElementById('text').innerHTML = str_new;
}

// circuit with single gate for now
var forwardMultiplyGate = function(x, y) { return x * y; };

// ----------------------------------------------------------
// STRATEGY #1: RANDOM SEARCH
// ----------------------------------------------------------
var x = -2, y = 3; // some input values
// try changing x,y randomly small amounts and keep track of what works best
var tweak_amount = 0.01;
var best_out = -Infinity;
var best_x = x, best_y = y;
for(var k = 0; k < 100; k++) {
  var x_try = x + tweak_amount * (Math.random() * 2 - 1); // tweak x a bit
  var y_try = y + tweak_amount * (Math.random() * 2 - 1); // tweak y a bit
  var out = forwardMultiplyGate(x_try, y_try);
  if(out > best_out) {
    // best improvement yet! Keep track of the x and y
    best_out = out; 
    best_x = x_try, best_y = y_try;
  }
}
print("Best x: " + best_x);
print("Best y: " + best_x);
print("Result: " + best_out);

// ----------------------------------------------------------
// STRATEGY #2: PARTIAL DERIVATIVES --> GRADIENT
// ----------------------------------------------------------
// Based on following function: 
// \frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
var x = -2, y = 3;
var out = forwardMultiplyGate(x, y); // -6
var h = 0.0001; //small change in variable to measure change in function
// In theory we would want the gradient, ie the limit of the expression
// as h --> 0
// The gradient with respect to all inputs is a vector of all the partial derivatives
// The gradient is the direction of the steepest increase of the function

// compute derivative with respect to x
var xph = x + h; // -1.9999
var out2 = forwardMultiplyGate(xph, y); // -5.9997
var x_derivative = (out2 - out) / h; // 3.0

// compute derivative with respect to y
var yph = y + h; // 3.0001
var out3 = forwardMultiplyGate(x, yph); // -6.0002
var y_derivative = (out3 - out) / h; // -2.0

// ----------------------------------------------------------
// UPDATE BASED ON DERIVATIVES
// ----------------------------------------------------------
var step_size = 0.01;
var out = forwardMultiplyGate(x, y); // before: -6
x += step_size * x_derivative; // x becomes -1.97
y += step_size * y_derivative; // y becomes 2.98
var out_new = forwardMultiplyGate(x, y); // -5.87! exciting.
print(" Output based on full partial deriv: " + out_new);

// ----------------------------------------------------------
// STRATEGY #3: ANALYTIC GRADIENT
// ----------------------------------------------------------
/* 
Previously we analyzed the change in function output once for every input we have. Complexity of evaluating the gradient is linear in number of inputs. In practice, not feasible. Instead, we derive a direct expression, an analytic gradient. 
Plugging our expression into the definition of the derivative of y, we get:
\frac{\partial f(x,y)}{\partial x} = \frac{f(x+h,y) - f(x,y)}{h}
= \frac{(x+h)y - xy}{h}
= \frac{xy + hy - xy}{h}
= \frac{hy}{h}
= y
The analytic gradient of x is y, and for y it is x! We can simplify our derivative calculation.
*/
var x = -2, y = 3;
var out = forwardMultiplyGate(x, y); // before: -6
var x_gradient = y; // by our complex mathematical derivation above
var y_gradient = x;

var step_size = 0.01;
x += step_size * x_gradient; // -2.03
y += step_size * y_gradient; // 2.98
var out_new = forwardMultiplyGate(x, y); // -5.87. Higher output! Nice.
print("Analytic gradient output: " + out_new);






