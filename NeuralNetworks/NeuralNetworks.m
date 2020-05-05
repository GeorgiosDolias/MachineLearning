classdef NeuralNetworks
    properties
        X = [];
        y = [];
        data = [];
        m = 1;
        iterations = 1000;
        lambda = 1;
        theta = []        
        input_layer_size  = 1; 
        num_labels = 1;
        hidden_layer_size = 25;   % 25 hidden units
                         
    end
    methods
        %Constructor
        function NN = NeuralNetworks(string)
            structure = load(string);
       
            NN.X = structure.X;
            NN.y = structure.y;
            NN.m = size(NN.X, 1);
            
            if size(NN.X,1) ~= size(NN.y,1)
                error('Invalid imported data');
            end
            NN.input_layer_size = size(NN.X,2);
            NN.num_labels = length(unique(NN.y));
            NN.theta = zeros(NN.num_labels, size(NN.X, 2)+ 1);           
        end
        
        %   Randomly selects 100 data points to display.
        %   Displays 2D data in X in a nice grid stored.
        function DisplayData(NN,example_width)
           
            rand_indices = randperm(NN.m);
            sel = NN.X(rand_indices(1:100), :);

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
        
        % Loads some pre-initialized neural network parameters.
        function [Theta1, Theta2] = LoadWeights(NN, string)
            structure = load(string);
       
            Theta1 = structure.Theta1;
            Theta2 = structure.Theta2;
        end
        %computes the sigmoid of z(z can be a matrix,vector or scalar).
        function g = Sigmoid(NN, z)
            g = zeros(size(z));
            g = 1./(1+ exp(-z));
        end
        
        %   Run through the examples one at the a time to 
        %   see what it is predicting.
        function Predictions(NN,Theta1, Theta2, X, y)
            m = size(X, 1); 
            %  Randomly permute examples
            rp = randperm(m);

            for i = 1:m
                % Display 
                fprintf('\nDisplaying Example Image\n');
                displayData(X(rp(i), :));

                pred = TrainAccuracy(NN,Theta1, Theta2, X(rp(i),:),y(rp(i)));
                fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

                % Pause with quit option
                s = input('Paused - press enter to continue, q to exit:','s');
                if s == 'q'
                  break
                end
            end
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