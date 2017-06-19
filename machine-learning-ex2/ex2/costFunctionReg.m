function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);
J_0 = -(y(1)*log(h(1)) + (1-y(1))'*log(1-h(1)))/m;
J_others = - (  y(2:end)' *log( h(2:end) ) ...
           + (1-y(2:end))'*log(1- h(2:end) ))/m ...
           + sum(theta(2:end) .^ 2)*lambda/(2*m);
J = J_0 + J_others;

grad_standard = (X'*(h-y)/m);
grad_0 = grad_standard(1);

grad_reg = grad_standard + lambda*theta/m;
grad_others = grad_reg(2:end);

grad = [grad_0; grad_others];
% =============================================================

end
