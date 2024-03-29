function visualizeBoundarySVM(X, y, modelsvm)
%VISUALIZEBOUNDARY plots a decision boundary learned by the SVM

% Plot the training data on top of the boundary
plotData(X, y)

% Make classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   [vals(:, i)] = svmpredict(zeros(size(this_X, 1), 1), this_X, modelsvm);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0 0], 'Color', 'b');
hold off;

end
