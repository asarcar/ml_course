function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, input_layer_size, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
% size(Theta1) = [hidden_layer_size input_layer_size+1]
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

% size(Theta2) = [num_labels hidden_layer_size+1]
Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
% J and grad 

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part 1
% ------
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

% size(Theta2) = [K l2+1] => size(z3) = [m K]
% K # of units in 3rd (final/output) layer = num_labels
z3 = a2*Theta2';
% htheta = a3 = output = g(z3) => size(htheta) = [m K]
htheta = sigmoid(z3);

% Generate Output Matrix yo (size [m K]) from y
% The row i of yo has a row vector with the column j
% 1 if y(i) = j
yo = (y == 1);
for label_value = 2:num_labels
    yo = [yo (y == label_value)];
end

% Element wise cost function
elem_J = (yo.*log(htheta)) + ((1-yo).*log(1-htheta));

% Remember Bias Weights of Theta1 and Theta2 that should NOT
% contribute to regularization.
Theta1_bias_v = Theta1(:,1);
Theta2_bias_v = Theta2(:,1);

% J(theta) = 
% -(1/m)*[for_all_training_units_i[
%           for_all_output_units_k(
%             y(i,k)log(htheta(i, k) + 
%             (1 - y(i,k))*(1 -log(htheta(i, k)))
%           )]] 
% +(lambda/2*m)[for_all_layers_except_output_layer_l[
%                 for_all_units_in_layer_l_j(
%                   for_all_units_in_layer_l+1_i(
%                     theta(i,j)^2
%                 ))]]
%
% -------------------------------------------------------------
J =  ( -(1/m)*(ones(1,m)*elem_J*ones(num_labels,1)) ) +             \
     ( (lambda/(2*m)) * ( dot(nn_params, nn_params) -               \
                          dot(Theta1_bias_v, Theta1_bias_v) -       \
                          dot(Theta2_bias_v, Theta2_bias_v) ) );
% =========================================================================
% 1. DELTA Matrix
%    for all training examples
%    1.a. Forward Propagation (OP): a_1 => z_2 => a_2 => z_2 => a_3
%    1.b. Backward Propagation (Error):
%         Compute Error (delta) at each node (except ip): 
%           1.b.x. OP Error: delta_3 = a_3 - y_o => 
%           1.b.y. Hidden Layer: 
%                  delta_2 = (Theta_2' * delta_3) .* d(g(z_2))/d(z_2)
%    1.c. Accumulate DELTA for each layer l=1:L-1
%         DELTA_l (for l=1:2) = DELTA_l +  (delta_l+1)*a_l'
% 2. d(J(theta))/d(theta): 
%    2.a. Average DELTA for each layer l=1:L-1
%         Theta_l_grad = (1/m)*DELTA_l
%    2.b. Regularize DELTA (except bias weights)
%         Theta_l_grad += (lambda/m).*Theta_l(:,2:end)
%    2.c. Unroll Theta_l_grad for all layers l=1:L-1
% -------------------------------------------------------------

% -------------------------------------------------------------
% 1. DELTA Matrix
%    for all training examples
%    1.a. Forward Propagation (OP): a_1 => z_2 => a_2 => z_2 => a_3

% a3 = htheta = a_3 computed for all training examples
% size(a3) = [m K] 

%    1.b. Backward Propagation (Error):
%         Compute Error (delta) at each node (except ip): 
%           1.b.x. OP Error: delta_3 = a_3 - y_o => 

% delta3 = delta_3 computed for all training examples: size(delta3) = [m K]
delta3 = htheta - yo; 

%           1.b.y. Hidden Layer: 
%                  delta_2 = (Theta_2' * delta_3) .* d(g(z_2))/d(z_2)

% delta2 = delta_2 computed for all training examples
% size(z2) = [m l2] = z_2 computed for all training examples
% size(Theta2(:,2:end)) = [K l2]
% size(delta2) = [m l2]; size(delta3) = [m K]
delta2 = (delta3*Theta2(:,2:end)) .* sigmoidGradient(z2);

%    1.c. Accumulate DELTA for each layer l=1:L-1
%         DELTA_l (for l=1:2) = DELTA_l +  (delta_l+1)*a_l'
% 2. d(J(theta))/d(theta): 
%    2.a. Average DELTA for each layer l=1:L-1
%         Theta_l_grad = (1/m)*DELTA_l
%    2.b. Regularize DELTA (except bias weights)
%         Theta_l_grad += (lambda/m).*Theta_l(:,2:end)

% Theta2_grad: computed for all training sets
% size(Theta2_grad) = [K l2+1]
% size(a2) = [m l2+1]; size(delta3') = [K m]
Theta2_grad = (1/m)*(delta3'*a2) +                                  \
              (lambda/m)*[zeros(num_labels,1) Theta2(:,2:end)];

% Theta1_grad: computed for all training sets
% size(Theta1_grad) = [l2 l1+1]
% size(a1) = [m l1+1]; size(delta2') = [l2 m]
Theta1_grad = (1/m)*(delta2'*a1) +                                  \
              (lambda/m)*[zeros(hidden_layer_size,1) Theta1(:,2:end)];

%    2.c. Unroll Theta_l_grad for all layers l=1:L-1
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

% -------------------------------------------------------------
% =========================================================================

end
