params = [0.01 0.03 0.1 0.3 1 3 10 30];

validation_scores = zeros(length(params), length(params));

for i = 1:length(params);
  for j = 1:length(params);
    C = params(i)
    sigma = params(j)
    model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(predictions ~= yval)
    validation_scores(i, j) = error;
   end
 end
 
 validation_scores

    