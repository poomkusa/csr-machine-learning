function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0;
sigma = 0;

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
minError = 99999;
c_test = 0.01;
while c_test<30
    sig_test = 0.01;
    while sig_test<30
        model = svmTrain(X, y, c_test, @(x1, x2) gaussianKernel(x1, x2, sig_test));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error < minError
            minError = error;
            C = c_test;
            sigma = sig_test;
        end
        sig_test = sig_test*3;
    end
    c_test = c_test*3;
end



% =========================================================================

end
