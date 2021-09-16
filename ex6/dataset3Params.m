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

C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_list = [0.01 0.03 0.1 0.3 1 3 10 30];

min_error = -100;
best_C = C_list(1);
best_sigma = sigma_list(1);

% Train on training set!!!
% Evaluate error on cross validation set!!!

for curr_C = C_list
    for curr_sigma = sigma_list
        model = svmTrain(X, y, curr_C, @(x1, x2)gaussianKernel(x1, x2, curr_sigma));
        predictions = svmPredict(model, Xval);
        prediction_error = mean(double(predictions ~= yval));
        if prediction_error ~= 0
            if min_error == -100
               min_error = prediction_error;
            elseif prediction_error < min_error
               min_error = prediction_error;
               best_C = curr_C;
               best_sigma = curr_sigma;
            end
        end
    end
end

C = best_C;
sigma = best_sigma;




% =========================================================================

end
