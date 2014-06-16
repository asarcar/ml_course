function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
C_vals = [1]';
num_Cs = length(C_vals);
sigma_vals = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]';
sigma_vals = [0.1]';
num_sigs = length(sigma_vals);

% Create "short hand" for the svmTrain function for various C/sigma
% svmTrainFnCall =                                                    \
%   @(C_val, sigma_val)                                               \
%   svmTrain(X, y, C_val, ...
% 	     @(x1, x2) gaussianKernel(x1, x2, sigma_val));
%
% Use libsvm
svmTrainFnCall = @(C_val, sigma_val)                                  \
                 svmTrainFn(X, y, 0, 2, C_val, 1/(2*sigma_val^2));

% Store the prediction values for various C and sigma values in a matrix
PredVals = zeros(num_Cs, num_sigs);

for cidx = 1:num_Cs
  for sidx = 1:num_sigs
    model = svmTrainFnCall(C_vals(cidx), sigma_vals(sidx));
    % PredV = svmPredict(model, Xval);
    % Use libsvm
    yval_norm = yval; yval_norm(find(yval_norm == 0)) = -1;
    [PredV] = svmpredict(yval_norm, Xval, model);
    % PredVals(cidx, sidx) = mean(double(PredV ~= yval_norm));
    % PredV
    % PredVals
  end 
end

% Return the C_vals and sigma_val with minimum prediction
% errors
[min_err, idx] = min(PredVals(:));
cidx = mod(idx-1,num_Cs) + 1;
sidx = ceil(idx/num_Cs);

% PredVals
% cidx
% sidx

C = C_vals(cidx);
sigma = sigma_vals(sidx);
% =========================================================================

end
