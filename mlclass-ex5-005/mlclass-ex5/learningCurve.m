function [error_train, error_val, error_test] = ...
    learningCurve(X, y, Xval, yval, Xtest, ytest, lambda)
%LEARNINGCURVE Generates the train, cross validation, and test 
% set errors needed to plot a learning curve
%   [error_train, error_val, error_test] = ...
%       LEARNINGCURVE(X, y, Xval, yval, Xtest, ytest, lambda) 
%       returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns three vectors of the same length - error_train, 
%       error_val, and error_test. 
%       Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i), error_test(i)).
%
%   In this function, you will compute the train, cv, and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
mval = size(Xval, 1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);
error_test  = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return training errors in 
%               error_train, cross validation errors in error_val, 
%               and test errors in error_test.
%               i.e., error_train(i), error_val(i), and error_test(i) 
%               should give you the errors
%               obtained after training on i examples.
%
% Note: You should evaluate the training error on the first i training
%       examples (i.e., X(1:i, :) and y(1:i)).
%
%       For the cross-validation, and test error, you should instead 
%       evaluate on the _entire_ set (Xval/Xtest and yval/ytest).
%
% Note: If you are using your cost function (linearRegCostFunction)
%       to compute the training and cross validation error, you should 
%       call the function with the lambda argument set to 0. 
%       Do note that you will still need to use lambda when running
%       the training to obtain the theta parameters.
%
% Hint: You can loop over the examples with the following:
%
%       for i = 1:m
%           % Compute train/cross validation errors using training examples 
%           % X(1:i, :) and y(1:i), storing the result in 
%           % error_train(i), error_val(i), and error_test(i)
%           ....
%           
%       end
%

% ---------------------- Sample Solution ----------------------

for i = 1:m

    % Compute train/cross validation errors using training examples 
    % X(1:i, :) and y(1:i), storing the result in 
    % error_train(i), error_val(i), and error_test(i)

    % For the training & cross validation error computation:
    % Average across multiple sets of randomly selected examples to 
    % determine the training and cross validation error. The average
    % is done accross a pre-determined set of trials
    % For the test error computation:
    % Average but computed accross the entire data for test 
    num_trials = 50;
    err_tmp_train = zeros(num_trials, 1);
    err_tmp_val   = zeros(num_trials, 1);
    err_tmp_test  = zeros(num_trials, 1);

    for j = 1:num_trials

      % Pick the first "i" rows of a random permutation of 
      % X to get Xtrain_permute
      rp_train       = randperm(m)(1:i);
      Xtrain_permute = X(rp_train, :);
      ytrain_permute = y(rp_train, :);
      [theta] = trainLinearReg(Xtrain_permute, ytrain_permute,      \
			       lambda);

      % Evaluate the training error on the i training examples 
      % Pass lambda = 0 ensures that unregularized cost is returned 
      [err_tmp_train(j), dummy] =                                   \
      linearRegCostFunction(Xtrain_permute, ytrain_permute, theta, 0);

      % Use the average of random permutation "i" entries assuming 
      % enough entries are available for cross validation
      if (i <= mval)
	 rp_val = randperm(m)(1:i);
	 Xval_permute = Xval(rp_val, :);
	 yval_permute = yval(rp_val, :);
	 [err_tmp_val(j), dummy] =                                  \
	 linearRegCostFunction(Xval_permute, yval_permute, theta, 0);
      else
	 [err_tmp_val(j), dummy] =                                  \
	 linearRegCostFunction(Xval, yval, theta, 0);
      end

      [err_tmp_test(j), dummy] =                                    \
      linearRegCostFunction(Xtest, ytest, theta, 0);
      
    end
      
    error_train(i) = mean(err_tmp_train);
    error_val(i)   = mean(err_tmp_val);
    error_test(i)  = mean(err_tmp_test);

end




% -------------------------------------------------------------

% =========================================================================

end
