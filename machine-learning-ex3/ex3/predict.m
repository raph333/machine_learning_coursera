function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

X_aug = [ones(m, 1) X];  % m x 401

% hidden layer features:
A_2 = sigmoid( Theta1 * X_aug' );  % 25 x m
A_2_aug = [ones(1, m); A_2];  % 26 x m

% output layer:
A_3 = sigmoid( Theta2 * A_2_aug );  % 10 x m


% get predictions from output layer:
[max_values, max_prob_class] = max(A_3, [], 1);
% max_prob_class: for each column (x): class with highest value (probability)
p = max_prob_class';
% return class (digit) with highest probability as prediction



% =========================================================================


end
