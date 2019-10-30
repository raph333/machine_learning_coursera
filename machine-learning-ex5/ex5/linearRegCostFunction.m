function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y);  % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

#{
predictions = X * theta;
errors = predictions - y;
square_errors = errors .** 2;
sum_of_square_errors  = ones(m, 1)' * square_errors;
%sum_of_square_errors = errors' * errors
J = sum_of_square_errors / (2*m);
%disp('J old:')
%J

%reg_term_grad = (lambda / m) * [0; theta(2:end)];  # don't regularize theta-0
%grad = ( (X*theta - y)' * X * (1/m) )' + reg_term_grad;
%disp('gradient old:')
%grad
#}





% note: X already contains first column of ones

mse = (1 / (2*m)) * ( (X*theta - y)' * (X*theta - y) );
reg_term = (lambda / (2*m)) * (theta(2:end)' * theta(2:end));
J = mse + reg_term;

gradient = (1 / m) * ( X' * (X*theta - y) );
reg_term_grad = (lambda / m) * [0; theta(2:end)];  # don't regularize theta-0
grad = gradient + reg_term_grad;


% =========================================================================

grad = grad(:);

end
