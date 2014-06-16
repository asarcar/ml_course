function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Set m: # of data points (row vectors)
m = size(X,1);

% You need to return the following variables correctly.
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% sq_dist_c: measure of distance of data points to centroids
% sq_dist_c(i, j): square of distance from x_i to centroid_j
sq_dist_c = zeros(m, K);
for ridx = 1:m
  for cidx = 1:K
    tmp_v = X(ridx, :)' - centroids(cidx, :)';
    sq_dist_c(ridx, cidx) = dot(tmp_v, tmp_v);
  end
end

[dummy, idx] = min(sq_dist_c, [], 2);

% =============================================================

end

