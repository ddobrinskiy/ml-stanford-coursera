function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])

%centroids=initial_centroids

% disable automatic broadcasting warning
warning ("off", "Octave:broadcast");
% Set K

K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               cloupdasest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.

% initalize structure to store distances from examples to all centroids
%X = X(1:5, :)
X_distances = [];
for j = 1:K
    % K = num of centroids;

    c = centroids(j, :);
    d = sum((X - c).^2, 2);
    % iterate over each centroid adding a column to X_distances
    % the new column contains m distances from x_i to current centroid_j
    X_distances = [X_distances, d];
end

% Now X_distances contains distances from X_i (a row) to each of K centroids
% (in the K columns).
% Now we need to assign the closest centroid to each X_i, which is the index
% of the minimum value row-wise
[min_val, idx] = min(X_distances, [], 2);
% =============================================================
end
