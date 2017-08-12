function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

% diff = vector column of diffirences between elements in x1 and x2
diff = x1-x2;
% distance - eucledian distance between x1 and x2
% distance = sum of diffirences between elements of x1 and x2
% implementation note: V*V' is equal to sum of squares of vector V;
%   V'*V is used instead of sum(V.^2) for PERFOMANCE reasons
distance = diff'*diff;
sim = exp(-distance/(2*sigma^2));
% =============================================================
end
