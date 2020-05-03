classdef MultiVariableLinearRegression
    properties
        X = [];
        y = [];
        data = [];
        m = 1;
        iterations = 1000
        alpha = 0.1; %     Learning rate
        theta = []
    end
    methods
        %Constructor
        function MLR = MultiVariableLinearRegression(string)
            MLR.data = load(string);
            if size(MLR.data,2) >= 2
                MLR.X = MLR.data(:, 1: end -1);
                MLR.y = MLR.data(:, end);
                MLR.m = length(MLR.y);
                MLR.theta = zeros(size(MLR.data,2), 1);
                if size(MLR.X,1) ~= size(MLR.y,1)
                    error('Invalid imported data');
                end
            else
                error('Invalid imported data');
            end
        end
         
        %   returns a normalized version of X where
        %   the mean value of each feature is 0 and the standard deviation
        %   is 1. This is often a good preprocessing step to do when
        %   working with learning algorithms.
        function X_Norm = FeatureNormalise(MLR)
            % You need to set these values correctly
            X_Norm = MLR.X;
            
            for i=1:size(MLR.X, 2)
   
               mean1 = mean(MLR.X(:,i));
               X_Norm(:,i)= X_Norm(:,i)-mean1;
               sig = std(X_Norm(:,i));
               X_Norm(:,i)= X_Norm(:,i)/sig;
            end
        end
        %   Add intercept term to X
        function extendedX = AddIntercept(MLR,matrix)
            extendedX = [ones(MLR.m, 1) matrix];
        end
        %   Computes the cost of using theta as the parameter
        %   for linear regression to fit the data points in X and y
        function Jcost = ComputeCostMulti(MLR,extX,Theta)
            Jcost = 0;
            sum = 0; 
            h=extX*Theta;

            for i=1:MLR.m
                sum= sum + (h(i)-MLR.y(i))^2;
            end
            
            Jcost = (1/(2*MLR.m))*sum;
        end
        %   Performs gradient descent to learn theta.
        %   It updates theta by taking iterations gradient steps 
        %   with learning rate alpha
        function [Theta, J_history] = GradientDescentMulti(MLR, extX,Theta)
            J_history = zeros(MLR.iterations, 1);
            
            %Theta = zeros(size(MLR.data, 2), 1);
            for iter = 1:MLR.iterations
                
                h=extX*Theta;

                for j=1:size(extX,2)      % Features = Columns
                    
                    Theta(j) = Theta(j) - MLR.alpha*(1/MLR.m)*...
                                sum((h-MLR.y) .* extX(:,j));

                end
                
                % Save the cost J in every iteration    
                J_history(iter) = MLR.ComputeCostMulti(extX, Theta);

            end
        end
        %Plot convergence graph
        function PlotCostMulti(MLR,J_values)
            
            figure;
            plot(1:numel(J_values), J_values, '-b', 'LineWidth', 2);
            xlabel('Number of iterations');
            ylabel('Cost J');
            title('Values of Cost function with respect to iterations')
        end
        % Predict values for various population sizes
        function price = Predictions(MLR, X_norm, values, Theta)
            
            mu = mean(X_norm);
            sigma = std(X_norm);
            val = values.*mu./sigma;
            price = [1 val] *Theta;
            fprintf('Predicted price of a %5.2f  sq-ft house ',values(1))
            fprintf('with %2.0f  bedrooms ',values(2))
            fprintf(' (using gradient descent): %f\n', price);
        end
        %   Computes the closed-form solution to linear regression. 
        %   using the normal equations.
        function theta =  NormalEqn(MLR,extX)
            theta = inv(extX'*extX)*extX'*MLR.y;
        end
        % Predict values for various population sizes based on normal eq.
        function price = NormPredictions(MLR, values, Theta)
            
            price = [1 values] *Theta;
            fprintf('Predicted price of a %5.2f , %5.2f ',values(1),values(2))
            fprintf(' (using normal equations): %f\n', price);
        end
    end
end