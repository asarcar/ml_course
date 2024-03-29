%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');

% m = Number of examples
m = size(X, 1);

% Plot training data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
%

theta = [1 ; 1];
J = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 3: Regularized Linear Regression Gradient =============
%  You should now implement the gradient for regularized linear 
%  regression.
%

theta = [1 ; 1];
[J, grad] = linearRegCostFunction([ones(m, 1) X], y, theta, 1);

fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a great 
%                 fit.
%

%  Train linear regression with lambda = 0
lambda = 0;
[theta] = trainLinearReg([ones(m, 1) X], y, lambda);

%  Plot fit over the data
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
hold off;

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
%

lambda = 0;
[error_train, error_val, error_test] = ...
    learningCurve([ones(m, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  [ones(size(Xtest, 1), 1) Xtest], ytest, ...
                  lambda);

plot(1:m, error_train, 1:m, error_val, 1:m, error_test);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation', 'Test')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training\tTrain\t\tCV\t\tTest Error\n');
for i = 1:m
    fprintf('  \t%d\t%f\t%f\t%f\n', i,                              \
            error_train(i), error_val(i), error_test(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%

p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize
X_poly = [ones(m, 1), X_poly];                   % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
X_poly_val = bsxfun(@minus, X_poly_val, mu);
X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
X_poly_test = bsxfun(@minus, X_poly_test, mu);
X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%
lambda_v = [0 1 100];
for i=1:length(lambda_v)
    lambda = lambda_v(i);
    [theta] = trainLinearReg(X_poly, y, lambda);

    % Plot training data and fit
    figure(2*i-1);
    plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
    plotFit(min(X), max(X), mu, sigma, theta, p);
    xlabel('Change in water level (x)');
    ylabel('Water flowing out of the dam (y)');
    title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));
    
    figure(2*i);
    [error_train, error_val, error_test] = ...
    learningCurve(X_poly, y, ...
		  X_poly_val, yval, ...
		  X_poly_test, ytest, lambda);
    plot(1:m, error_train, 1:m, error_val, 1:m, error_test);
    
    title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
    xlabel('Number of training examples')
    ylabel('Error')
    axis([0 13 0 100])
    legend('Train', 'Cross Validation', 'Test')
    fprintf('Polynomial Regression: Lambda = %f:\n', lambda);
    fprintf('   Min/Max Train Vs CV Vs Test Error ');
    fprintf('(%f/%f), (%f/%f), (%f/%f)\n\n',                        \
	    min(error_train), max(error_train),                     \
	    min(error_val), max(error_val),                         \
            min(error_test), max(error_test)); 
end

fprintf('Program paused. Press enter to continue.\n');
pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val, error_test] = ...
    validationCurve(X_poly, y, X_poly_val, yval, X_poly_test, ytest);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val, lambda_vec, error_test);
legend('Train', 'Cross Validation', 'Test')
xlabel('lambda');
ylabel('Error');
% set(gca, 'XTick', [1:(length(lambda_vec)-1)]);
% set(gca, 'XTicklabel',                                            \
%     {'0.001', '0.003', '0.01', '0.03',                            \
%      '0.1', '0.3', '1', '3', '10'});
set(gca, 'XTick',                                                   \
    [floor([min(lambda_vec)]):1:ceil([max(lambda_vec)])]);
set(gca, 'YTick',                                                   \
    [floor([min(error_train), min(error_val), min(error_test)]):2:  \
     ceil([max(error_train), max(error_val), max(error_test)])]);
fprintf('lambda\t\tTrain\t\tValidation\tTest Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i),                          \
            error_val(i), error_test(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;

% Get a better indication of performance in the real
% world, it is important to model on a test set that was
% not used in any part of training (that is, it was neither 
% used to select the lambda parameters nor the model
% parameters.
close all;

fprintf('Regularized Polynomial Regression: Lambda Data\n');
for i = 1:length(lambda_vec)
    theta = trainLinearReg(X_poly, y, lambda_vec(i));
    % Evaluate the error realized on test data
    fprintf('\tLambda = %f: Error on CV %f/Test %f\n',              \
            lambda_vec(i), error_val(i), error_test(i)); 
end

% Get the "best" lambda that contributes minimum error_val 
fprintf('Regularized Polynomial Regression Results:\n');
[dummy_min, min_idx] = min(error_val);
fprintf('\tCV Based Best Lambda = %f\n', lambda_vec(min_idx)); 
fprintf('\tError on Test Data = %f\n', error_test(min_idx));
[dummy_min, min_idx] = min(error_test);
fprintf('\tTest Based Best Lambda = %f\n', lambda_vec(min_idx)); 
fprintf('\tError on Test Data = %f\n', error_test(min_idx));


