classdef RegularisedLinearRegression
    properties
        X = [];
        y = [];
        Xtest = [];
        ytest = [];
        Xval = [];
        yval = [];
        
        data = [];
        m = 1;
        lambda = 1;
        theta = []
    end
    methods
        %Constructor
        function RLR = RegularisedLinearRegression(string)
            if string ~= ""
                structure = load(string);

                RLR.X = structure.X;
                RLR.y = structure.y;
                RLR.Xtest = structure.Xtest;
                RLR.ytest = structure.ytest;
                RLR.Xval = structure.Xval;
                RLR.yval = structure.yval;

                RLR.m = size(RLR.X, 1);
                if size(RLR.X,1) ~= size(RLR.y,1)
                    error('Invalid imported data');
                end
            end
        end
        %   Plots the data points X and y into a new figure 
        %   It plots the data points with + for the positive examples
        %   and o for  negative examples. X is assumed to be a Mx2 matrix.
        function PlotData(RLR,X,y)
            % m = Number of examples
            m = size(X, 1);

            % Plot training data
            plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
            xlabel('Change in water level (x)');
            ylabel('Water flowing out of the dam (y)');
        end
         
        %   Computes cost and gradient for logistic regression with 
        %   regularization Computes the cost of using theta as the 
        %   parameter for regularized logistic regression and the
        %   gradient of the cost w.r.t. to the parameters. 
        function [J,grad] = LinearRegCostFunction(RLR,X, y, theta, lambda)
            % Initialize some useful values
            m = length(y); % number of training examples

            % You need to return the following variables correctly 
            J = 0;
            grad = zeros(size(theta));

            h = X*theta;
            theta_reg = [0;theta(2:end, :);];
            J = (1/(2*m)) * sum((h - y).^2) + (lambda/(2*m)) *...
                (theta_reg' * theta_reg);
            grad = (1/m) * X' * (h - y) + (lambda/m) * theta_reg;
            
            grad = grad(:);
        end
        %   Trains linear regression given a dataset (X, y) and a
        %   regularization parameter lambda       
        %   Returns the trained parameters theta.
        function [theta] = TrainLinearReg(RLR,X, y, lambda)
            % Initialize Theta
            initial_theta = zeros(size(X, 2), 1); 

            % Create "short hand" for the cost function to be minimized
            costFunction = @(t) linearRegCostFunction(X, y, t, lambda);

            % Now, costFunction is a function that takes in only one argument
            options = optimset('MaxIter', 200, 'GradObj', 'on');

            % Minimize using fmincg
            theta = fmincg(costFunction, initial_theta, options);
        end
        
       %  Plot fit over the data
       function PlotFit(RLR, X, y, theta)
            % m = Number of examples
            m = size(X, 1);
            %  Plot fit over the data
            plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
            xlabel('Change in water level (x)');
            ylabel('Water flowing out of the dam (y)');
            hold on;
            plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2)
            hold off;
       end
       
        %   Generates the train and cross validation set errors needed 
        %   to plot a learning curve
        %   Returns the train and cross validation set errors for a
        %   learning curve. In particular, it returns two vectors of 
        %   the same length- error_train and error_val.
        %   Then, error_train(i) contains the training error for
        %   i examples (and similarly for error_val(i)).
       function [error_train, error_val] = LearningCurve(RLR,X, y, Xval,...
                                            yval, lambda)
            % Number of training examples
            m = size(X, 1);

            % You need to return these values correctly
            error_train = zeros(m, 1);
            error_val   = zeros(m, 1);
        
            for i= 1:m
                theta = trainLinearReg(X(1:i,:),y(1:i),lambda);
                
                error_train(i)=ComputeCost(RLR,X(1:i,:),y(1:i),theta);

                Jv= computeCost(Xval,yval,theta);
                error_val(i) = Jv;
            end

            
            plot(1:m, error_train, 1:m, error_val);
            title('Learning curve for linear regression')
            legend('Train', 'Cross Validation')
            xlabel('Number of training examples')
            ylabel('Error')
            axis([0 13 0 150])

            fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
            for i = 1:m
                fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
            end
       end
       
       %   Compute cost for linear regression
       %   Computes the cost of using theta as the
       %   parameter for linear regression to fit the data points 
       %   in X and y
       function J = ComputeCost(RLR,X, y, theta)
            
            m = length(y); % number of training examples

            J = 0;

            sum = 0;
            h=X*theta;

            for i=1:m,
                sum= sum + (h(i)-y(i))^2;
            end

            J = (1/(2*m))*sum;

       end       
       
       %   Maps X (1D vector) into the p-th power
       %   It takes a data matrix X (size m x 1) and
       %   maps each example into its polynomial features where
       %   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
       function [X_poly] = PolyFeatures(RLR,X, p)
       
            X_poly = zeros(numel(X), p);

            X_poly(:,1)= X;

            for i=2:p 
               X_poly(:,i) = X.*X_poly(:,i-1); 
            end
       end 
       
       %   Normalizes features in X. 
       %   It returns a normalized version of X where
       %   the mean value of each feature is 0 and the standard deviation
       %   is 1. This is often a good preprocessing step to do when
       %   working with learning algorithms.
       function [X_norm, mu, sigma] = FeatureNormalize(RLR,X)
            
            mu = mean(X);
            X_norm = bsxfun(@minus, X, mu);

            sigma = std(X_norm);
            X_norm = bsxfun(@rdivide, X_norm, sigma);

       end 
       
       function [X_poly,X_poly_val,mu, sigma] = FeatureMapping(RLR,X,Xtest,Xval)
           m = size(X, 1);

           p = 8;

           % Map X onto Polynomial Features and Normalize
           X_poly = PolyFeatures(RLR,X, p);
           [X_poly, mu, sigma] = FeatureNormalize(RLR,X_poly);  % Normalize
           X_poly = [ones(m, 1), X_poly];                   % Add Ones

            % Map X_poly_test and normalize (using mu and sigma)
            X_poly_test = PolyFeatures(RLR,Xtest, p);
            X_poly_test = bsxfun(@minus, X_poly_test, mu);
            X_poly_test = bsxfun(@rdivide, X_poly_test, sigma);
            X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

            % Map X_poly_val and normalize (using mu and sigma)
            X_poly_val = PolyFeatures(RLR,Xval, p);
            X_poly_val = bsxfun(@minus, X_poly_val, mu);
            X_poly_val = bsxfun(@rdivide, X_poly_val, sigma);
            X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

            fprintf('Normalized Training Example 1:\n');
            fprintf('  %f  \n', X_poly(1, :));
           
       end
       
       %   Experiment with polynomial regression with multiple
        %  values of lambda. Try running the code with different values of
        %  lambda to see how the fit and learning curve change.
       function [err_train, err_val] = TestLearningCurve(RLR, X, X_poly_val ,X_poly,y,yval,...
                            mu, sigma,error_train,error_val,lambda)
            
             m = size(X, 1);
             
             p=8;
            
             [theta] = TrainLinearReg(RLR,X_poly, y, lambda);
            
             
            % Plot training data and fit
            figure(1);
            plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
            plotFit(min(X), max(X), mu, sigma, theta, p);
            xlabel('Change in water level (x)');
            ylabel('Water flowing out of the dam (y)');
            title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

            figure(2);
            [err_train, err_val] = ...
                LearningCurve(RLR,X_poly, y, X_poly_val, yval, lambda);
            plot(1:m, error_train, 1:m, error_val);

            title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
            xlabel('Number of training examples')
            ylabel('Error')
            axis([0 13 0 100])
            legend('Train', 'Cross Validation')

            fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
            fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
            for i = 1:m
                fprintf('  \t%d\t\t%f\t%f\n', i, err_train(i), err_val(i));
            end
       end
       
       %    Generates the train and validation errors needed to
       %    plot a validation curve that we can use to select lambda.
       %    Returns the train and validation errors (in error_train, error_val)
       %    for different values of lambda. 
       function ValidationCurve(RLR, X_poly, X_poly_val,y, yval)
       
           % Selected values of lambda (you should not change this)
            lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
            m = size(X_poly, 1);              
            
            % You need to return these variables correctly.
            error_train = zeros(length(lambda_vec), 1);
            error_val = zeros(length(lambda_vec), 1);
            
            for i = 1:length(lambda_vec)
               lambda = lambda_vec(i);
               theta = trainLinearReg(X_poly,y,lambda);

               error_train(i) = ComputeCost(RLR,X_poly, y, theta);
               error_val(i) = ComputeCost(RLR,X_poly_val, yval, theta);
            end
            
            close all;
            plot(lambda_vec, error_train, lambda_vec, error_val);
            legend('Train', 'Cross Validation');
            xlabel('lambda');
            ylabel('Error');

            fprintf('lambda\t\tTrain Error\tValidation Error\n');
            for i = 1:length(lambda_vec)
                fprintf(' %f\t%f\t%f\n', ...
                        lambda_vec(i), error_train(i), error_val(i));
            end           
       end        
    end
end