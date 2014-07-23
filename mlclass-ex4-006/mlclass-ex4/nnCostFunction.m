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
            X=[ones(m,1) X];%X扩展一列
            hidden1=sigmoid(X * Theta1');%计算第一隐藏层
            hidden1=[ones(m,1) hidden1];%对第一隐藏层进行扩展
            output=sigmoid(hidden1*Theta2');%输出层
            yout=zeros(m,1);
            for i=1:num_labels
                yout=(y==i);
                J=J+sum(-1*yout'*log(output(:,i))-(1-yout')*(log(1-output(:,i))))/m;
            end               
            Theta1_Re=Theta1(:,2:end);
            Theta2_Re=Theta2(:,2:end);
            J=J+(sum(sum(Theta1_Re .^2))+sum(sum(Theta2_Re.^2)))*lambda/(2*m);%正则化
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
                a2=ones(1,hidden_layer_size+1);
            for i=1:m
                a1=X(i,:);%第i个测试样例,1x401
                z2=a1 * Theta1';%1x25
                a2(2:end)=sigmoid(z2);%1x26
                z3=a2 * Theta2';%1x10
                a3=sigmoid(z3);%1x10        
                yReal=zeros(1,num_labels);%1x10
                yReal(y(i))=1;
                delta3=a3-yReal;%1x10
                delta2=delta3*(Theta2(:,2:end)).*sigmoidGradient(z2);%1x25
                Theta2_grad=Theta2_grad+delta3' * a2;
                Theta1_grad=Theta1_grad+delta2' * a1;                                
            end
             Theta2_grad=Theta2_grad ./m;
             Theta1_grad=Theta1_grad ./m;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.

% -------------------------------------------------------------
             theta2_tmp=zeros(size(Theta2));
             theta1_tmp=zeros(size(Theta1));
             theta2_tmp(:,2:end)=Theta2(:,2:end);
             theta1_tmp(:,2:end)=Theta1(:,2:end);
             Theta2_grad=Theta2_grad +theta2_tmp*lambda/m;
             Theta1_grad=Theta1_grad +theta1_tmp*lambda/m;           
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
