classdef NeuralNetworksLearning
    properties
        X = [];
        y = [];
        data = [];
        m = 1;
        theta = []        
        input_layer_size  = 1; 
        num_labels = 1;
        hidden_layer_size = 25;   % 25 hidden units
                         
    end
    methods
        %Constructor
        function NNL = NeuralNetworksLearning(string)
            if string ~= ""
                structure = load(string);

                NNL.X = structure.X;
                NNL.y = structure.y;
                NNL.m = size(NNL.X, 1);

                if size(NNL.X,1) ~= size(NNL.y,1)
                    error('Invalid imported data');
                end
                NNL.input_layer_size = size(NNL.X,2);
                NNL.num_labels = length(unique(NNL.y));
                NNL.theta = zeros(NNL.num_labels, size(NNL.X, 2)+ 1);
            end
        end
        
        %   Randomly selects 100 data points to display.
        %   Displays 2D data in X in a nice grid stored.
        function DisplayData(NNL,X,example_width)
           

            % Set example_width automatically if not passed in
            if ~exist('example_width', 'var') || isempty(example_width) 
                example_width = round(sqrt(size(X, 2)));
            end

            % Gray Image
            colormap(gray);

            % Compute rows, cols
            [k, q] = size(X);
            size(example_width);
            example_height = (q / example_width);

            % Compute number of items to display
            display_rows = floor(sqrt(k));
            display_cols = ceil(k / display_rows);

            % Between images padding
            pad = 1;

            % Setup blank display
            display_array = - ones(pad + display_rows * (example_height + pad), ...
                                   pad + display_cols * (example_width + pad));

            % Copy each example into a patch on the display array
            curr_ex = 1;
            for j = 1:display_rows
                for i = 1:display_cols
                    if curr_ex > k 
                        break; 
                    end
                    % Copy the patch

                    % Get the max value of the patch
                    max_val = max(abs(X(curr_ex, :)));
                    display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                                  pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                                    reshape(X(curr_ex, :), example_height, example_width) / max_val;
                    curr_ex = curr_ex + 1;
                end
                if curr_ex > k 
                    break; 
                end
            end

            % Display Image
            h = imagesc(display_array, [-1 1]);

            % Do not show axis
            axis image off

            drawnow;
        end
        
        % Loads some pre-initialized neural network parameters.
        function nn_params = LoadWeights(NNL, string)
            structure = load(string);
       
            Theta1 = structure.Theta1;
            Theta2 = structure.Theta2;
            
            % Unroll parameters 
            nn_params = [Theta1(:) ; Theta2(:)];
        end
        %   Computes the sigmoid of z(z can be a matrix,vector or scalar).
        function g = SigmoidGradient(NNL, z)
            g = zeros(size(z));
            g = sigmoid(z).*(1-sigmoid(z));
        end
        %   Implements the neural network cost function for a two layer
        %   neural network which performs classification
        %   Computes the cost and gradient of the neural network. The
        %   parameters for the neural network are "unrolled" into the vector
        %   nn_params and need to be converted back into the weight matrices. 
        % 
        %   The returned parameter grad should be a "unrolled" vector of the
        %   partial derivatives of the neural network.
        function [Jcost, grad] = ComputeCost(NNL,nn_params, ...
              input_layer_size,hidden_layer_size, num_labels, X, y, lambda) 
            
            % Reshape nn_params back into the parameters Theta1 and Theta2,
            % the weight matrice for our 2 layer neural network
           Theta1 = reshape(nn_params(1:hidden_layer_size * ...
          (input_layer_size + 1)),hidden_layer_size,(input_layer_size + 1));

            Theta2 = reshape(nn_params((1 + (hidden_layer_size *...
         (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));
            
            % Setup some useful variables
            m = size(X, 1);
            K = num_labels;
            X= [ones(m, 1) X];  
            
            % Initialise returned values 
            Jcost = 0;
            Theta1_grad = zeros(size(Theta1));
            Theta2_grad = zeros(size(Theta2));
            
            %   Step 1 
            all_combos = eye(num_labels);    
            y_matrix = all_combos(y,:) ; 
            size(y_matrix);
            
            %   Step 2
            a1=X;
            a2 = sigmoid(a1 * Theta1');
            h = sigmoid([ones(m, 1) a2] * Theta2');
            
            %   Step 3a-> Summations for non-regularized cost function
            
            sumi=0;
            for i=1:m
                sumk=0;

                for k=1:K
                    sumk = sumk + (-y_matrix(i,k)*log(h(i,k))-(1-y_matrix(i,k))*log(1-h(i,k)));        
                end
                sumi= sumi + sumk;
            end

            %Step 3b-> Summations for Theta

            %Sum for Theta1
            sumj1=0;

            for j=1:size(Theta1, 1);
                sumj1 = sumj1 + sum(Theta1(j,2:end).^2); 
            end


            %Sum for Theta2
            sumj2=0;
            for j=1:size(Theta2, 1);
                sumj2 = sumj2 + sum(Theta2(j,2:end).^2); 
            end
            
            %Step 3c-> Regularized cost function

            Jcost = (1/m)*sumi+(lambda/(2*m))*(sumj1+sumj2);
            
            % Step 4 Backpropagation algorithm

            D1= zeros(size(Theta1));
            D2=zeros(size(Theta2));

            for t=1:m

                a1=X(t,:);
                size(a1);

                z2=a1 * Theta1';
                a2 = sigmoid(z2);
                size(a2);
                a2 = [ones(size(a2,1), 1) a2];
                size(a2);

                z3 = a2* Theta2';    
                a3 = sigmoid(z3);

                size(a3);


                d3 = h(t,:)-y_matrix(t,:);
                size(d3);
                size(Theta2(:,2:end)');

                d2 =(d3*Theta2(:,2:end)).*sigmoidGradient(z2);
                size(d2);

                size(d3'*a2);

                D2 =D2 + d3'*a2;
                size(D2);
                D1 =D1 + d2'*a1;
                size(D1);
            end
            
            Theta1_grad = (1/m)*D1;
            Theta2_grad = (1/m)*D2;

            Theta1_grad(:,2:end) =Theta1_grad(:,2:end) + (lambda/m)*...
                Theta1(:,2:end);
            Theta2_grad(:,2:end) =Theta2_grad(:,2:end) + (lambda/m)*...
                Theta2(:,2:end);
            
            % Unroll gradients
            grad = [Theta1_grad(:) ; Theta2_grad(:)];
            
        end
        
        %   Randomly initialize the weights of a layer with L_in
        %   incoming connections and L_out outgoing connections. 
        function W = randInitializeWeights(NNL,L_in, L_out)
            W = zeros(L_out, 1 + L_in);
            epsilon_init = 0.12;
            W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
        end
        
        %   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
        %   backpropagation gradients, it will output the analytical gradients
        %   produced by your backprop code and the numerical gradients (computed
        %   using computeNumericalGradient). These two gradient computations should
        %   result in very similar values.
        function checkNNGradients(NNL,lambda)
            if ~exist('lambda', 'var') || isempty(lambda)
                lambda = 0;
            end

            input_layer_size = 3;
            hidden_layer_size = 5;
            num_labels = 3;
            m = 5;

            % We generate some 'random' test data
            Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size);
            Theta2 = debugInitializeWeights(num_labels, hidden_layer_size);
            % Reusing debugInitializeWeights to generate X
            X  = debugInitializeWeights(m, input_layer_size - 1);
            y  = 1 + mod(1:m, num_labels)';

            % Unroll parameters
            nn_params = [Theta1(:) ; Theta2(:)];

            % Short hand for cost function
            costFunc = @(p) nnCostFunction(p, input_layer_size, hidden_layer_size, ...
                                           num_labels, X, y, lambda);

            [cost, grad] = costFunc(nn_params);
            numgrad = computeNumericalGradient(costFunc, nn_params);

            % Visually examine the two gradient computations.  The two columns
            % you get should be very similar. 
            disp([numgrad grad]);
            fprintf(['The above two columns you get should be very similar.\n' ...
                     '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

            % Evaluate the norm of the difference between two solutions.  
            % If you have a correct implementation, and assuming you used EPSILON = 0.0001 
            % in computeNumericalGradient.m, then diff below should be less than 1e-9
            diff = norm(numgrad-grad)/norm(numgrad+grad);

            fprintf(['If your backpropagation implementation is correct, then \n' ...
                     'the relative difference will be small (less than 1e-9). \n' ...
                     '\nRelative Difference: %g\n'], diff);

      
        end
        function [Theta1,Theta2] = TrainingNN(NNL, initial_nn_params, ...
                                   input_layer_size,hidden_layer_size, ...
                                   num_labels, X, y, lambda)
            %  change the MaxIter to a larger value to see how more 
            %   training helps.
            options = optimset('MaxIter', 50);                   
            % Create "short hand" for the cost function to be minimized
            costFunction = @(p) ComputeCost(NNL,p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);
            % Now, costFunction is a function that takes in only one argument (the
            % neural network parameters)
            [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
            
            % Obtain Theta1 and Theta2 back from nn_params
            Theta1 = reshape(nn_params(1:hidden_layer_size * ... 
         (input_layer_size + 1)),hidden_layer_size, (input_layer_size + 1));

            Theta2 = reshape(nn_params((1 + (hidden_layer_size * ...
         (input_layer_size + 1))):end),num_labels, (hidden_layer_size + 1));

        end
        
        
        
        
        
        
        
        
        %   Predicts the label of an input given a trained neural network.
        %   Outputs the predicted label of X given the
        %   trained weights of a neural network (Theta1, Theta2)

        function pred = Predictions(NNL,Theta1, Theta2, X,y)
            % Useful values
            m = size(X, 1);
            num_labels = size(Theta2, 1);

            % You need to return the following variables correctly 
            pred = zeros(size(X, 1), 1);

            h1 = sigmoid([ones(m, 1) X] * Theta1');
            h2 = sigmoid([ones(m, 1) h1] * Theta2');
            [dummy, pred] = max(h2, [], 2);
            fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
        end
        
       
        %  After training the neural network, we would like to use it to predict
        %  the labels. You will now implement the "predict" function to use the
        %  neural network to predict the labels of the training set. This lets
        %  you compute the training set accuracy.
        function pred = TrainAccuracy(NN,Theta1,Theta2, X, y)
            m = size(X, 1);
            num_labels = size(Theta2, 1);

            % You need to return the following variables correctly 
            pred = zeros(size(X, 1), 1);

            % Add ones to the X data matrix
            X = [ones(m, 1) X];
            
            a1= sigmoid(X*Theta1');
            a1 = [ones(size(a1, 1), 1) a1];
            %qsize(a1)
            h= sigmoid(a1*Theta2');
            %h = X*all_theta';
            [M, I]= max(h, [],2);

            pred=I;
            
            fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
        end
    end
end