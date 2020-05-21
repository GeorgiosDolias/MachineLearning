classdef OneVsAllLogisticRegression
    properties
        X = [];
        y = [];
        data = [];
        m = 1;
        lambda = 1;
        theta = []        
        input_layer_size  = 1; 
        num_labels = 1;          
                         
    end
    methods
        %Constructor
        function OVALR = OneVsAllLogisticRegression(string)
            if string ~= ""
                structure = load(string);

                OVALR.X = structure.X;
                OVALR.y = structure.y;
                OVALR.m = size(OVALR.X, 1);

                if size(OVALR.X,1) ~= size(OVALR.y,1)
                    error('Invalid imported data');
                end
                OVALR.input_layer_size = size(OVALR.X,2);
                OVALR.num_labels = length(unique(OVALR.y));
                OVALR.theta = zeros(OVALR.num_labels, size(OVALR.X, 2)+ 1);
            end
        end
        
        %   Randomly selects 100 data points to display.
        %   Displays 2D data in X in a nice grid stored.
        function DisplayData(OVALR,example_width)
           
            rand_indices = randperm(OVALR.m);
            sel = OVALR.X(rand_indices(1:100), :);

            % Set example_width automatically if not passed in
            if ~exist('example_width', 'var') || isempty(example_width) 
                example_width = round(sqrt(size(sel, 2)));
            end

            % Gray Image
            colormap(gray);

            % Compute rows, cols
            [k, q] = size(sel);
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
                    max_val = max(abs(sel(curr_ex, :)));
                    display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
                                  pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
                                    reshape(sel(curr_ex, :), example_height, example_width) / max_val;
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
     
        %computes the sigmoid of z(z can be a matrix,vector or scalar).
        function g = Sigmoid(OVALR, z)
            g = zeros(size(z));
            g = 1./(1+ exp(-z));
        end
         
        %   Computes cost and gradient for logistic regression with 
        %   regularization. Computes the cost of using
        %   theta as the parameter for regularized logistic regression 
        %   and the gradient of the cost w.r.t. to the parameters. 
        function [J,grad] = LRComputeCost(OVALR,newTheta,X,y,lambda)
            J = 0;   
            m = length(y);
            sum2=0;

            h = sigmoid(X*newTheta);

            sum2 =sum(newTheta(2:length(newTheta)).^2);
            
            % Vectorized implementation

            J = (1/m)*(-y'*log(h)-(1-y)'*log(1-h))+ (lambda/(2*m))*sum2;            

            % Vectorized Gradient

            %sum3 = 
            %sum3
            grad =(1/m)* X'*(h-y);

            temp= newTheta;
            temp(1) = 0;

            grad = grad + (lambda/m)*temp;  
            
            grad = grad(:);
            
        end
        %   Trains multiple logistic regression classifiers and returns all
        %   the classifiers in a matrix all_theta, where the i-th row 
        %   of all_theta corresponds to the classifier for label i.
        %   Trains num_labels logistic regression classifiers and returns 
        %   each of these classifiers in a matrix all_theta, where the 
        %   i-th row of all_theta corresponds to the classifier for label i
        function [all_theta] = oneVsAll(OVALR, y, num_labels, lambda)
            
            % Some useful variables
            m = size(X, 1);
            n = size(X, 2);

            % You need to return the following variables correctly 
            all_theta = zeros(num_labels, n + 1);

            % Add ones to the X data matrix
            X = [ones(m, 1) X];
            
            
            for c=1:num_labels
    
                % Initialize fitting parameters
                initial_theta = zeros(n + 1, 1);

                %  Set options for fminunc
                options = optimset('GradObj', 'on', 'MaxIter', 50);

                %  Run fminunc to obtain the optimal theta
                %  This function will return theta and the cost 
                [theta] = ...
                      fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                             initial_theta, options);
                theta         
                all_theta(c,:) = theta; 
            end
        end
        %  Predicts the label for a trained one-vs-all classifier. The labels 
        %  are in the range 1..K, where K = size(all_theta, 1). 
        %  It will return a vector of predictions for each example in the 
        %  matrix X. Note that X contains the examples in
        %  rows. all_theta is a matrix where the i-th row is a trained logistic
        %  regression theta vector for the i-th class. You should set p to a vector
        %  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
        %  for 4 examples) 
        function pred = predictOneVsAll(OVALR,all_theta, X, y)
            m = size(X, 1);
            num_labels = size(all_theta, 1);

            % You need to return the following variables correctly 
            p = zeros(size(X, 1), 1);

            % Add ones to the X data matrix
            X = [ones(m, 1) X];
            
            h = X*all_theta';
            [M I]= max(h, [],2);

            pred=I;
            
            fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
        end
    end
end