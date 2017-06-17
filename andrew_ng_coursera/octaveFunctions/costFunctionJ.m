function J = costFunctionJ(X, y, theta)

% X is the "design matrix" containing training examples
% y is the class labels

m = size(X,1);    % number of training examples
predictions = X*theta; % predictions of hypthesis on all m examples
sqrErrors = (predictions - y).^2; % squared errors by-element

J = 1/(2*m) * sum(sqrErrors); % return cost function