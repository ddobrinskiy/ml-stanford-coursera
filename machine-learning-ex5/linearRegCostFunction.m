function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% y_predicted of theta
h = X*theta;
err_squared = (y - h).^2;

J_linreg = sum(err_squared)/(2*m);

% first param not regularized --> exclude theta_1
reg_penalty = lambda*sum( theta(2:end).^2 )/(2*m);
J = J_linreg + reg_penalty;
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)

% grad = (unregularized gradient for logistic regression)
grad = X'*(h-y)/m;
temp = theta;
temp(1) = 0;   % because we don't add anything for j = 0
% grad = grad + YOUR_CODE_HERE (using the temp variable)
grad = grad + lambda*temp/m;






% =========================================================================

grad = grad(:);

end
