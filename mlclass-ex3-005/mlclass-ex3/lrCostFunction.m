function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
% element wise sigmoid function: m size vector
htheta = sigmoid(X*theta);   
			    
% theta_reg operates on all theta weights except the bias (1st) element
theta_reg = theta; theta_reg(1) = 0; 

% J = scalar_value = Sum(for all i) 
% [-y_i * log(htheta_i) - (1 - y_i)*(1 - log(htheta_i)]/m + lambda*theta_i^2/2*m  
J = ((-(y'*log(htheta) + (1 - y')*log(1 - htheta))) + (lambda*dot(theta_reg, theta_reg)/2))/m;

% grad = n + 1 vector:
% grad_j = Sum(for all i)
% [x_i_j*(htheta_i - y_i)/m + lambda*theta_reg_j/m]
grad = ((X'*(htheta - y)) + (lambda*theta_reg))/m;
% =============================================================

end
