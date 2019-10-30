% Take from:
% https://swizec.com/blog/i-suck-at-implementing-neural-networks-in-octave/swizec/2929


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
 
yy = zeros(size(y),num_labels);
for i=1:size(X)
  yy(i,y(i)) = 1;
end
 
X = [ones(m,1) X];
% cost
for  i=1:m
  a1 = X(i,:);
  z2 = Theta1*a1';
  a2 = sigmoid(z2);
  z3 = Theta2*[1; a2];
  a3 = sigmoid(z3);
 
  J += -yy(i,:)*log(a3)-(1-yy(i,:))*log(1-a3);
end
 
J /= m;
 
J += (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));
 
t=1;
for t=1:m
  % forward pass
  a1 = X(t,:);
  z2 = Theta1*a1';
  a2 = [1; sigmoid(z2)];
  z3 = Theta2*a2;
  a3 = sigmoid(z3);
 
  % backprop
  delta3 = a3-yy(t,:)';
  delta2 = (Theta2'*delta3).*[1; sigmoidGradient(z2)];
  delta2 = delta2(2:end);
  
  #{
  disp('delta2:')  
  size(delta2)
  
  disp('update D1 and D2:')
  disp("delta3 * a2'")
  size(delta3 * a2')
  
  disp('Theta2_grad:')
  size(Theta2_grad)
 #}
 
  Theta1_grad = Theta1_grad + delta2*a1;
  Theta2_grad = Theta2_grad + delta3*a2';
end
 
Theta1_grad = (1/m)*Theta1_grad+(lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m)*Theta2_grad+(lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];
 
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
 
end
