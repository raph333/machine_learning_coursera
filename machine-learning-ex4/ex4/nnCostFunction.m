function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

#{
disp('Theta1 and Theta2')
size(Theta1)
size(Theta2)
#}    



% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

# CALCULATE COST:
# Note: Here, A_2 and A_3 are structured like X: one sample - row.
# In general, it is more convenient to have A1, A2, ... in the Format
# one column - one sample

# first, get predictions:
X_aug = [ones(m, 1) X];  % m x 401
%size(X_aug)

% hidden layer features:
A_2 = sigmoid( X_aug * Theta1' );  % m x 25
%size(A_2)
A_2_aug = [ones(m, 1) A_2];  % m x 26
%size(A_2_aug)

% output layer:
A_3 = sigmoid( A_2_aug * Theta2' );  % m x 10
%size(A_3)
# A_3: matrix of predictions of training examples
# row: prediction of one observation
# - e.g.: [0.25 0.92 0.08 0.05 0.02 0.06 0.30 0.20 0.05 0.02]

# now: calcluate cost:
y_onehot = ( y == ones(m, 1) * 1:num_labels );
# row: one-hot endoded label of one observation- e.g.: [0 0 1 0 0 0 0 0 0 0]

errors = -(y_onehot .* log(A_3)) - ((1 - y_onehot) .* log(1 - A_3));
J = sum(errors(:)) / m;

reg_term = (lambda / (2*m)) * ( sum( (Theta1(:,2:end) .**2 )(:) ) +
                                sum( (Theta2(:,2:end) .**2 )(:) ) );
J = J + reg_term;


# BACKPROPAGATION:

Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for t = 1:m;
  
  # feed forward one pattern:
  a1 = X(t,:)';  # column vector
  a1_aug = [1; a1];
  z2 = Theta1 * a1_aug;
  a2 = sigmoid(z2);
  a2_aug = [1; a2];
  z3 = Theta2 * a2_aug;
  a3 = sigmoid(z3);
  
  # backpropagate the pattern:
  y = y_onehot(t,:)';  # column vector [0,0,0,0,1,0,0,0,0,0]'
  delta3 = a3 - y;
  
  #{
  disp('delta3:')
  size(delta3)
  
  disp("Theta2' * delta3")
  size((Theta2' * delta3))
  disp("g'(z2)")  
  size(sigmoidGradient(z2))
  #}

  delta2 = (Theta2' * delta3)(2:end) .* sigmoidGradient(z2);

  #{
  disp('delta2:')  
  size(delta2)
  
  disp('update D1 and D2:')
  disp("delta3 * a2'")
  size(delta3 * a2')
  disp("delta2 * a1'")
  size(delta2 * a1')
  #}

  
  Delta2 = Delta2 + (delta3 * a2_aug');
  Delta1 = Delta1 + (delta2 * a1_aug');
  
end

Theta1_grad = (1 / m) * Delta1;
reg_term1 = (lambda / m) * Theta1;
reg_term1(:,1) = 0;  # don't regularize bias
Theta1_grad = Theta1_grad + reg_term1;

Theta2_grad = (1 / m) * Delta2;
reg_term2 = (lambda / m) * Theta2;
reg_term2(:,1) = 0;
Theta2_grad = Theta2_grad + reg_term2;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
