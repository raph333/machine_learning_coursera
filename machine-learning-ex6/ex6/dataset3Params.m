function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;  % just some initializations; does not matter
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

param_values = [0.01 0.03 0.1 0.3 1 3 10 30];
n_vals = length(param_values);
validation_errors = zeros(n_vals, n_vals);

for i = 1:n_vals;  % row: C values
  for j = 1:n_vals;  % columns: sigma values
    C = param_values(i)
    sigma = param_values(j)
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(predictions ~= yval)
    validation_errors(i, j) = error;
  end
end

validation_errors
minimal_error = min(min(validation_errors))
[row, column] = find(validation_errors == minimal_error)

C = param_values(row)  % best C value
sigma = param_values(column)  % best sigma value


% =========================================================================

end
