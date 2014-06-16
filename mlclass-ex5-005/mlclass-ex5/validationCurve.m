function [lambda_vec, error_train, error_val, error_test] = ...
    validationCurve(X, y, Xval, yval, Xtest, ytest)
%VALIDATIONCURVE Generate the train, validation, and test errors 
% needed to plot a validation curve that we can use to select 
% lambda
%   [lambda_vec, error_train, error_val, error_test] = ...
%       VALIDATIONCURVE(X, y, Xval, yval, Xtest, ytest) returns the train
%       validation, and test errors (in error_train, error_val, error_test)
%       for different values of lambda. You are given the training set (X,
%       y), validation set (Xval, yval), and test set (Xtest, ytest)
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);
error_test = zeros(length(lambda_vec), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train and the validation errors in error_val. The 
%               vector lambda_vec contains the different lambda parameters 
%               to use for each calculation of the errors, i.e, 
%               error_train(i), and error_val(i) should give 
%               you the errors obtained after training with 
%               lambda = lambda_vec(i)
%
% Note: You can loop over lambda_vec with the following:

for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  % Compute train / val errors when training linear 
  % regression with regularization parameter lambda
  % You should store the result in error_train(i)
  % and error_val(i)
  [err_train_tmp, err_val_tmp, err_test_tmp] =                      \
    learningCurve(X, y, Xval, yval, Xtest, ytest, lambda);
  error_train(i) = err_train_tmp(length(err_train_tmp));
  error_val(i) = err_val_tmp(length(err_val_tmp));
  error_test(i) = err_test_tmp(length(err_test_tmp));
end











% =========================================================================

end
