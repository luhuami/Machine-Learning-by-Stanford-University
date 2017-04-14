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
#sigma = 0.3;

sigma = 0.01;

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
test_C = C;
test_sigma = sigma;
error = 1000000;
for i = 1:10
  test_C = i * test_C;
  for j = 1:10
    test_sigma = j * test_sigma;
    model= svmTrain(X, y, test_C, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
    predictions = svmPredict(model, Xval);
    new_error = mean(double(predictions ~= yval))
    if( (i>1 || j>1) && new_error < error)
      error = new_error;
      C = test_C;
      sigma = test_sigma;
    endif
  end
end

% =========================================================================

end
