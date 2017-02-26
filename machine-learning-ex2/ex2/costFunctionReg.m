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

hypothesis = sigmoid(theta' * X');

c1 = (-(log(hypothesis) * y) - log(ones(1,m) - hypothesis)*(ones(m,1) - y))/m;

c2 = (lambda/(2*m)) * (sum(theta .^ 2) - theta(1,1) ^2);

J = c1 + c2;

[l,n] = size(X);

gradient = zeros(size(theta));
for iter = 1:n
  if(iter == 1)
    gradient(iter, 1) = ((hypothesis - y') * X(:, iter))/m;
  else
    gradient(iter, 1) = ((hypothesis - y') * X(:, iter))/m + (lambda * theta(iter,1)/m);
  endif
end

grad = gradient;




% =============================================================

end
