function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
param_grid = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

% maxtrix A will contain all parameters and resulting error score
% initialize empty A
A = []
counter=0
grid_len = size(param_grid)(2)

for i=1:grid_len
    for j=1:grid_len
        counter += 1;
        C = param_grid(i);
        sigma = param_grid(j);
        printf('======================================\n');
        printf('Iteration %d out of %d\n', counter, grid_len*grid_len);
        printf('C: %d\n', C);
        printf('sigma: %d\n', sigma);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        J = mean(double(predictions ~= yval));
        printf('J: %d\n', J);

        % add C, sigma, J as a new row to matrix A
        A = [A;[C sigma J]];
    end
end

% store index of the BEST pair of parameters in `iw`
[w, iw] = min(A(:, 3));

% unpack best parameters from iw's row of matrix A
C = A(iw, :)(1);
sigma = A(iw, :)(2);
J = A(iw, :)(3);



% =========================================================================

end
