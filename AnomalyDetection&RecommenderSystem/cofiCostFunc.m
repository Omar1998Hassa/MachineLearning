function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
X = reshape(params(1:num_movies*num_features),num_movies,num_features);
theta = reshape(params(1+num_movies*num_features:end),num_users,num_features);
%size(X)
%size(theta)
%size(Y(R))
h = X*theta';
J = 0.5 * sum(sum((h(R)-Y(R)).^2));  
%size(h(R)-Y(R))
%X_grad = (h(R)-Y(R))*theta();

%Theta_grad = (h(R)-Y(R))'*X;
%for i = 1 : num_movies
    
    %idx = find(R(i,:)==1);
   % Theta_temp = Theta(idx,:);
  %  Y_temp = Y(i,idx);
 %   X_grad(i,:) = (X(i,:)*Theta_temp'-Y_temp)*Theta_temp
 
%end
X_grad = ((X*theta'-Y).*R)*theta;
%(X*theta'-Y).*R
%R

%for i = 1 : num_users
   %idx = find(R(:,1)==1);
   % X_temp = X(idx,:);
   % Y_temp = Y(idx,i);
    %size(X_temp)
    %size(Y_temp)
   % %size(theta(i,:))
   % size(((X*theta'-Y).*R)'*X)
  %  ((X*theta'-Y).*R)'*X
 %   Theta_grad(i,:) = X_temp'*(X_temp*theta(i,:)'-Y_temp)
    
%end
Theta_grad=((X*theta'-Y).*R)'*X;
%((X*theta'-Y).*R)'

J = J +(lambda/2)*sum(sum(theta.^2))+(lambda/2)*sum(sum(X.^2))
X_grad= X_grad + lambda.*(X)
Theta_grad = Theta_grad + lambda*(theta)






% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
