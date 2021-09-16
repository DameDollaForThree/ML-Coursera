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
%

% one hot encoding y matrix
new_y = zeros( size(y,1), num_labels);  % initialize new_y to be 5000 x 10 matrix
for i = 1:m
    new_y(i, y(i)) = 1;     % populate the new_y matrix
end

% feed forward
X = [ones(m, 1) X];   % add a '1' in the front of every row
h1 = sigmoid(X * Theta1');   % we get 5000 x 25
h1 = [ones(m,1) h1];   % we get 5000 x 26
result = sigmoid(h1 * Theta2');  % we get 5000 x 10

% regularization (remember to exclude the first column of bias)
regularization = (lambda / (2*m)) * ( sum(Theta1(:,2:end) .^2, 'all') + sum(Theta2(:,2:end) .^2, 'all') );

% compute cost
J = (1/m) * sum( (-new_y .* log(result) - (1-new_y) .* log(1-result)), 'all' ) + regularization;

% initialize deltas
delta_2 = zeros(size(Theta2));
delta_1 = zeros(size(Theta1));

% backprop with for loop
for t = 1:m
    % step 1: feedforward
    a_1 = X(t,:);     % 1 x 401 (remember to add ':' to include all columns)
    z_2 = a_1 * Theta1';    % 1 x 25
    a_2 = sigmoid(z_2);     % 1 x 25
    a_2 = [1 a_2];     % 1 x 26 (concatenation)
    z_3 = a_2 * Theta2';    % 1 x 10
    a_3 = sigmoid(z_3);     % 1 x 10
    
    % step 2: epsilon_3 for the output layer
    curr_y = new_y(t,:);    % remember to add ':' to include all columns
    epsilon_3 = a_3 - curr_y;   % 1 x 10
    
    % step 3: epsilon_2 for the hidden layer
    epsilon_2 = epsilon_3 * Theta2(:,2:end) .* sigmoidGradient(z_2);   % 1x25
    
    % step 4: accumulate the gradient
    delta_2 = delta_2 + epsilon_3' * a_2;     % 10x1 * 1x26 = 10x26
    delta_1 = delta_1 + epsilon_2' * a_1;    % 25x1 * 1x401 = 25x401
    
    
end

% step 5: final division: normalize the gradient and add regularization
Theta1_grad = delta_1 / m + (lambda/m)*[zeros(hidden_layer_size, 1) Theta1(:,2:end)];
Theta2_grad = delta_2 / m + (lambda/m)*[zeros(num_labels, 1) Theta2(:,2:end)];
    
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
