function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%

% for effecient calculation of cost function, let's iterate over Users
% because theta has Users x Features, X has Movies x Users
%  for j = 1:num_users
%      % loop iterates over USERS
%      usr_ix = R(:, j);
%      usr_theta = Theta(j, :);
%      %usr_theta = [0.4, 0.1, 1]
%      usr_rated_movies_features = X(usr_ix, :);
%      usr_y_pred = usr_rated_movies_features * usr_theta';
%      usr_y_val = Y(usr_ix, j);
%      % Calculate cost for current User
%      % usr_j = sum((usr_y_pred - usr_y_val).^ 2)
%      % below is a more effecient matrix implementations of usr_j
%      usr_err = usr_y_pred - usr_y_val;
%      usr_j = (usr_err'*usr_err)/2;
%
%      % add User cost to total Cost
%      J += usr_j;
%  end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Implementation Note: We strongly encourage you to use a vectorized
% implementation to compute J, since it will later by called many times
% by the optimization package fmincg. As usual, it might be easiest to
% first write a non-vectorized implementation (to make sure you have the
% right answer), and the modify it to become a vectorized implementation
% (checking that the vectorization steps don’t change your algorithm’s output).
% To come up with a vectorized implementation, the following tip
% might be helpful: You can use the R matrix to set selected entries to 0.
% For example, R .* M will do an element-wise multiplication between M
% and R; since R only has elements with values either 0 or 1, this has the
% effect of setting the elements of M to 0 only when the corresponding value
% in R is 0. Hence, sum(sum(R.*M)) is the sum of all the elements of M for
% which the corresponding element in R equals 1.

% cost without regularization
J_err = 1/2 * sum(sum(R.*((X*Theta' - Y).^2)));

% gradient without normalization
X_grad          =  R.*(X*Theta' - Y)   * Theta;
Theta_grad      = (R.*(X*Theta' - Y))' * X;


% regularization penalty
reg_pen = sum(sum(Theta .^ 2)) + sum(sum(X .^ 2));
J = J_err + reg_pen * lambda/2;


% gradient with normalization
X_grad     += lambda*X;
Theta_grad += lambda*Theta;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
