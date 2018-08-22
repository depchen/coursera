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

%向前传播
X=[ones(m,1),X];%5000*401
hidden1_net=X*Theta1';%5000*25
hidden1_out=sigmoid(hidden1_net);%5000*25
hidden1_out=[ones(m,1),hidden1_out];%5000*26
hidden2_net=hidden1_out*Theta2';%5000*10
hidden2_out=sigmoid(hidden2_net);%5000*10

for i=1:m
	labels=zeros(num_labels,1);%初始化labels为10*1的矩阵
	result=y(i);%拿出结果
	labels(result)=1;%结果位置设置为1
	J=J+log(hidden2_out(i,:))*(-labels)-log(1-hidden2_out(i,:))*(1-labels);%累加代价函数
	%向后传播
	%diff_output = hidden2_out(i, :)' - labels;%计算输出层差值   10*1
	%delta2 = Theta2(:, 2:end)' * diff_output .* sigmoidGradient(hidden1_net(i, :)');%26*1  隐藏层delta
		
	%delta1 = Theta1'*delta2 ; %401*1  delta
	%Theta2_grad = Theta2_grad + diff_output*hidden1_out(i, :);
    %Theta1_grad = Theta1_grad + delta2*X(i, :);
	%向后传播
	
    delta2 = hidden2_out(i, :)' - labels; %计算输出层差值  
    delta1 = Theta2(:, 2:end)' * delta2 .* sigmoidGradient(hidden1_net(i, :)');%计算隐藏层的差值
	
	Theta2_grad = Theta2_grad + delta2 * hidden1_out(i, :);
    Theta1_grad = Theta1_grad + delta1 * X(i, :);
end
%正则化
J = J / m;
Theta2_grad = Theta2_grad / m;
Theta1_grad = Theta1_grad / m;

Theta1_regular = [zeros(hidden_layer_size,1), Theta1(:, 2:end)];
Theta2_regular = [zeros(num_labels,1), Theta2(:, 2:end)];

J = J + (sum(sum(Theta1_regular.^2)) + sum(sum(Theta2_regular.^2))) * lambda/ 2 / m;
Theta1_grad = Theta1_grad + Theta1_regular * lambda / m;
Theta2_grad = Theta2_grad + Theta2_regular * lambda / m;










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
