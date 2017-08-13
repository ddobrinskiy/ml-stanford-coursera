function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));
[m n] = size(X);
% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X

% initialize matrix for random centroids
c_ix = [];
while size(c_ix, 1) < K
    rand_ix = ceil(rand()*m);
    c_ix = [c_ix; rand_ix];
end

centroids = X(c_ix, :);



% =============================================================

end
