% ------------------------------
% Elimentary Function J(x)
% Ax = b => computes square (|Ax - b|)
% ------------------------------
function J = linearCostFn(A, b, x)
% Return: J(x): return value
% Arg1: A training input data
% Arg2: b training output data
% Arg3: x predicted coefficients
m = size(A, 1) % # of rows: training data
p = A*x % predicted data
err = sum((p - b) .^2) % square(|Ax - b|)
J = 1/(2*m) * err; % Error Function Value
