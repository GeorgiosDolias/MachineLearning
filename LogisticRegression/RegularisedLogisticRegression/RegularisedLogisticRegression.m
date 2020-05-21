classdef RegularisedLogisticRegression
    properties
        X = [];
        y = [];
        data = [];
        m = 1;
        lambda = 1;
        theta = [];
    end
    methods
        %Constructor
        function RLR = RegularisedLogisticRegression(string)
            if string ~= ""
                RLR.data = load(string);
                if size(RLR.data,2) >= 2
                    RLR.X = RLR.data(:, 1: end -1);
                    RLR.y = RLR.data(:, end);
                    RLR.m = length(RLR.y);
                    RLR.theta = zeros(size(RLR.data,2), 1);
                    if size(RLR.X,1) ~= size(RLR.y,1)
                        error('Invalid imported data');
                    end
                else
                    error('Invalid imported data');
                end
            end
        end
        
        % Plot training data
        function PlotData(RLR,X)
            % Create New Figure
            figure; hold on;
            
            % Find indices of Positive and negative examples
            pos = find(RLR.y==1); neg = find(RLR.y==0);

            % Plot Examples
            plot(X(pos,1),X(pos,2),'k+','LineWidth',2,...
                    'MarkerSize',7);
            plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','c',...
                    'MarkerSize',7);
            hold off;
            
            % Put some labels 
            hold on;
            % Labels and Legend
            xlabel('Microchip Test 1')
            ylabel('Microchip Test 2')

            % Specified in plot order
            legend('y = 1', 'y = 0')
            hold off;
        end
        %   In some cases, a dataset might have data points that are not
        %   linearly separable. However, you would still like to use 
        %   logistic regression to classify the data points.
        %
        %   To do so, you introduce more features to use -- in particular, 
        %   you add polynomial features to our data matrix (similar to
        %   polynomialregression).
        %   Maps the two input features to quadratic features 
        %   used in the regularization exercise.
        %   Returns a new feature array with more features, comprising of 
        %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

        %   Inputs X1, X2 must be the same size
        function featured_map = MapFeature(RLR,col1,col2)
            degree = 6;
            featured_map = ones(size(RLR.X(:,1)));
            for i = 1:degree
                for j = 0:i
                    featured_map(:, end+1) = (col1.^(i-j)).*(col2.^j);
                end
            end
        end
        function init_theta = InitialiseTheta(RLR,extX)
            % Initialize fitting parameters
            init_theta = zeros(size(extX, 2), 1);
        end
        %computes the sigmoid of z(z can be a matrix,vector or scalar).
        function g = Sigmoid(RLR,z)
            g = zeros(size(z));
            g = 1./(1+ exp(-z));
        end
         
        %   Compute cost and gradient for regularized linear 
        %   regression with multiple variables;
        %   Computes the cost of using theta as the parameter 
        %   for linear regression to fit the data points in X and y. 
        %   Returns the cost in J and the gradient in grad
        function [J,grad] = ComputeCost(RLR,newX,newTheta,lambda)
            % Initialize some useful values
            m = length(y); % number of training examples

            % You need to return the following variables correctly 
            J = 0;
            grad = zeros(size(theta));

            % ====================== YOUR CODE HERE ======================
            % Instructions: Compute the cost and gradient of regularized linear 
            %               regression for a particular choice of theta.
            %
            %               You should set J to the cost and grad to the gradient.
            %

            h = X*theta;
            sum =0;


            for j=2:length(theta);
                sum = sum + theta(j)^2;
            end

            sum3 =0;
            for i=1:m,
                sum3= sum3 + (h(i)-y(i))^2;
            end


            J = (1/(2*m))*sum3+ (lambda/(2*m))*sum;



            for j=1:size(theta,1);
                sum2=0;
                %j
                for i=1:m;
                    sum2 = sum2 + (h(i)-y(i))*X(i,j);
                end
                if(j==1);
                    grad(j) = (1/m)*sum2;
                else
                    grad(j) = (1/m)*sum2 + lambda*theta(j)/m;
                end
            end





            % =========================================================================

            grad = grad(:);
        end
        %   use a built-in function (fminunc) to find the
        %   optimal parameters theta.
        function [optimal_theta, cost,lambda] = OptimalParameters(RLR,extX,initial_theta,lambda)
            %  Set options for fminunc
            options = optimset('GradObj', 'on', 'MaxIter', 400);

            %  Run fminunc to obtain the optimal theta
            %  This function will return theta and the cost 
            [optimal_theta, cost, exit_flag] = ...
            fminunc(@(t)(costFunctionReg(t, extX, RLR.y, lambda)), initial_theta, options);
        end
        %Plot boundary
        function PlotBoundary(RLR,extX,opt_theta,lambda)
            % Plot Data
            RLR.PlotData(RLR.X);
            hold on

            if size(extX, 2) <= 3
                % Only need 2 points to define a line, so choose two endpoints
                plot_x = [min(extX(:,2))-2,  max(extX(:,2))+2];

                % Calculate the decision boundary line
                plot_y = (-1./opt_theta(3)).*(opt_theta(2).*plot_x + opt_theta(1));

                % Plot, and adjust axes for better viewing
                plot(plot_x, plot_y)

                % Legend, specific for the exercise
                legend('Admitted', 'Not admitted', 'Decision Boundary')
                axis([30, 100, 30, 100])
            else
                % Here is the grid range
                u = linspace(-1, 1.5, 50);
                v = linspace(-1, 1.5, 50);

                z = zeros(length(u), length(v));
                % Evaluate z = theta*x over the grid
                for i = 1:length(u)
                    for j = 1:length(v)
                        z(i,j) = mapFeature(u(i), v(j))*opt_theta;
                    end
                end
                z = z'; % important to transpose z before calling contour

                % Plot z = 0
                % Notice you need to specify the range [0, 0]
                contour(u, v, z, [0, 0], 'LineWidth', 2)
            end
            hold off
            
            % Put some labels 
            hold on;
            title(sprintf('lambda = %g', lambda))
            
            % Labels and Legend
            xlabel('Microchip Test 1')
            ylabel('Microchip Test 2')

            legend('y = 1', 'y = 0', 'Decision boundary')

            % Specified in plot order
            hold off;
        end
        
        %   Predict whether the label is 0 or 1 using learned logistic 
        %   regression parameters theta.
        %   It computes the predictions for X using a threshold at 0.5 (i.e.,
        %   if sigmoid(theta'*x) >= 0.5, predict 1)
        %   Compute accuracy on our training set
        function accuracy = TrainAccuracy(RLR,extX,opt_theta)
            predictions = zeros(RLR.m, 1);
            h = sigmoid(extX*opt_theta);

            for j=1:RLR.m
                if h(j)< 0.5
                    predictions(j) =0;
                else
                    predictions(j) = 1;
                end    
            end
            accuracy = mean(double(predictions == RLR.y)) * 100;
            fprintf('Train Accuracy: %f\n', accuracy);
        end
    end
end