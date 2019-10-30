function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.


for k = 1:K;
  assigned_data_point_indices = find(idx == k);
  assigned_data_points = X( assigned_data_point_indices, : );
  new_centroid = mean(assigned_data_points);  # row vector 1xn
  if (length(assigned_data_point_indices) == 0)
    new_centroid = X( randi([1, m]), :);  # assign centroid to random data point
  endif
  #size(new_centroid)
  #new_centroid
  centroids(k, :) = new_centroid;
 end


% =============================================================


end

