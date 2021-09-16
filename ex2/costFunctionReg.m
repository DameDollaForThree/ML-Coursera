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


prediction = sigmoid(X*theta);
% index slicing: use '(' not '['; use 'end' to denote the last element
regularization = (lambda/(2*m)) .* sum( theta(2:end) .^2 );
% use '.*' for element-wise multiplication
J = (1/m) .* sum( -y .* log(prediction) - (1-y) .* log(1-prediction) ) + regularization  ;

% extract the first element from all rows: use (:,1)
grad(1) = (1/m) * ( X(:,1)' * (prediction-y) );
grad(2:end) = (1/m) * ( X(:, 2:end)' * (prediction-y) ) + (lambda/m) * theta(2:end);



% =============================================================

end
