%% Machine Learning Online Class
%  Exercise 1: Linear regression with multiple variables
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression exercise. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     warmUpExercise.m       : Done in ex1.m
%     plotData.m             : Done in ex1.m
%     gradientDescent.m      : Done in ex1.m
%     computeCost.m          : 
%     gradientDescentMulti.m : 
%     computeCostMulti.m     : 
%     featureNormalize.m     : 
%     normalEqn.m            : 
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%

%% Initialization

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear ; close all; clc

fprintf('Loading data ...\n');

%% Load Data
data = load('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Print out some data points
fprintf('First 10 examples from the dataset: \n');
fprintf(' x = [%.0f %.0f], y = %.0f \n', [X(1:10,:) y(1:10,:)]');

fprintf('Program paused. Press enter to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma] = featureNormalize(X);

% Add intercept term to X
X = [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha). 
%
%               Your task is to first make sure that your functions - 
%               computeCost and gradientDescent already work with 
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%

fprintf('Running gradient descent ...\n');

% Choose different alpha values and monitor convergence of theta
% alphas are separated by a multiple of 3
alphav = [0.001 0.0033 0.01 0.033 0.1 0.33 0.5 0.75 0.9 1.0 1.1 1.2];
% Colors: blacK, Red, Green, Blue, Magenta, Cyan
colorsv = [1 3 5 2 4 6 1 3 5 2 4 6];
stylev  = ['-' '.' '^' '-' '.' '^' '-' '.' '^' '-' '.' '^'];
num_iters = 50;

figure;

for trial = 1:length(alphav)

  % Init Theta and Run Gradient Descent 
  alpha = alphav(trial);
  lineS = strcat(num2str(stylev(trial)), num2str(colorsv(trial)));
  theta = zeros(3, 1);
  [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);

  % Plot the convergence graph
  plot(1:numel(J_history), J_history, lineS, 'LineWidth', 1);
  hold on;

  % Display gradient descent's result
  fprintf('Theta computed from gradient descent (alpha %0.2f):\n', alpha);
  fprintf(' %0.2f \n\n', theta);

end

legend('alpha=0.001', 'alpha=0.0033', 'alpha=0.01', 'alpha=0.033', \
       'alpha=0.1', 'alpha=0.33', 'alpha=0.5', 'alpha=0.75', \
       'alpha=0.9', 'alpha=1.0', 'alpha-1.1', 'alpha-1.2'); 
title('Gradient Descent Convergence for different alpha values');
xlabel('Number of iterations');
set(gca, 'XTick', [1:2:num_iters]);
ylabel('Cost J');

hold off % don't overlay any more plots on this figure

% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = [1 ([1650 3] - mu)./sigma]*theta; % You should change this

% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================

fprintf('Solving with normal equations...\n');

% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in 
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a 1650 sq-ft, 3 br house.
%

%% Load Data
data = csvread('ex1data2.txt');
X = data(:, 1:2);
y = data(:, 3);
m = length(y);

% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
% ====================== YOUR CODE HERE ======================
price = [1 1650 3]*theta; % You should change this


% ============================================================

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

