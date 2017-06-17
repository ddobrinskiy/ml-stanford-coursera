function J = computeCostMulti(X, y, theta)
%COMPUTECOSTMULTI Compute cost for linear regression with multiple variables
%   J = COMPUTECOSTMULTI(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% compute PREDICTED values for given theta
h = X*theta;
% compute hypothesis errors for given theta
errors = h - y;
% compute squared errors (use .^ for vector element-wise power)
errors_squared = errors.^2;
% J = sum(errors_squared)/(2m)
J = sum(errors_squared)/(2*m);



% =========================================================================

end
