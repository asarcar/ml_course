function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Layer 1 Computation
% a1 = X + ones for bias term: size(a1) = [m l1+1] 
% m is # of training sets
% l1 # of units in input layer (layer 1) == # of features or dimension
a1 = [ones(m, 1) X];

% size(Theta1) = [l2 l1+1] => size(z2) = [m l2]
% l2 # of units in 2nd layer
z2 = a1*Theta1';

% Layer 2 Computation: size(a2) = [m l2+1]
% a2 = [1 g(z2)]
a2 = [ones(m, 1) sigmoid(z2)];

% size(Theta2) = [l3 l2+1] => size(z3) = [m l3]
% l3 # of units in 3rd (final/output) layer
z3 = a2*Theta2';
% htheta = a3 = output = g(z3)
htheta = sigmoid(z3);

% p would return a (m, 1) vector with indices (1, l3)
% note l3 is the number of output classes
[tmp_v, p] = max(htheta, [], 2);
% =========================================================================

end
