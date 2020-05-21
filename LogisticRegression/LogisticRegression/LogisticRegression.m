classdef LogisticRegression
    properties
        X = [];
        y = [];
        data = [];
        m = 1;     
        theta = []
    end
    methods
        %Constructor
        function LR = LogisticRegression(string)
            if string ~= ""
                LR.data = load(string);
                if size(LR.data,2) >= 2
                    LR.X = LR.data(:, 1: end -1);
                    LR.y = LR.data(:, end);
                    LR.m = length(LR.y);
                    LR.theta = zeros(size(LR.data,2), 1);
                    if size(LR.X,1) ~= size(LR.y,1)
                        error('Invalid imported data');
                    end
                else
                    error('Invalid imported data');
                end
            end
        end
        %    Plots the data points X and y into a new figure 
        %   It plots the data points with + for the positive examples
        %   and o for  negative examples. X is assumed to be a Mx2 matrix.

        function PlotData(LR,X)
            % Create New Figure
            figure; hold on;
            
            % Find indices of Positive and negative examples
            pos = find(LR.y==1); neg = find(LR.y==0);

            % Plot Examples
            plot(X(pos,1),X(pos,2),'k+','LineWidth',2,...
                    'MarkerSize',7);
            plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','c',...
                    'MarkerSize',7);
            hold off;
            
            % Put some labels 
            hold on;
            % Labels and Legend
            xlabel('Exam 1 score')
            ylabel('Exam 2 score')

            % Specified in plot order
            legend('Admitted', 'Not admitted')
            hold off;
        end
        %   Add intercept term to X
        function extendedX = AddIntercept(LR,matrix)
            extendedX = [ones(LR.m, 1) matrix];
        end
        %computes the sigmoid of z(z can be a matrix,vector or scalar).
        function g = Sigmoid(LR,z)
            g = zeros(size(z));
            g = 1./(1+ exp(-z));
        end
         
        %   Compute cost and gradient for logistic regression
        %   computes the cost of using theta as the
        %   parameter for logistic regression and the gradient of the cost
        %   w.r.t. to the parameters.
        function [J,grad] = ComputeCost(LR,newX,newTheta)
            J = 0;            
            grad = zeros(size(newTheta));
            h=sigmoid(newX*newTheta);
            
            J = (1/LR.m)*(-LR.y'*log(h)-(1-LR.y)'*log(1-h));
            

            for j=1:size(newTheta,1)
                sum=0;
                
                for i=1:LR.m
                    
                    sum = sum + (h(i)-LR.y(i))*newX(i,j);
                end
                grad(j) = (1/LR.m)*sum;
            end
        end
        %   use a built-in function (fminunc) to find the
        %   optimal parameters theta.
        function [optimal_theta, cost] = OptimalParameters(LR,extX)
            %  Set options for fminunc
            options = optimset('GradObj', 'on', 'MaxIter', 400);

            %  Run fminunc to obtain the optimal theta
            %  This function will return theta and the cost 
            [optimal_theta, cost] = ...
            fminunc(@(t)(costFunction(t, extX, LR.y)), LR.theta, options);
        end
        %Plot boundary that seperates points 
        function PlotBoundary(LR,extX,opt_theta)
            % Plot Data
            LR.PlotData(LR.X);
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
            % Labels and Legend
            xlabel('Exam 1 score')
            ylabel('Exam 2 score')

            % Specified in plot order
            legend('Admitted', 'Not admitted')
            hold off;
        end
        %   predict the outcomes on unseen data.
        function prob = Predictions(LR,params,theta)
            prob = LR.Sigmoid([1 params] * theta);
        end
        % Compute accuracy on our training set
        function accuracy = TrainAccuracy(LR,extX,opt_theta)
            predictions = zeros(LR.m, 1);
            h = sigmoid(extX*opt_theta);

            for j=1:LR.m
                if h(j)< 0.5
                    predictions(j) =0;
                else
                    predictions(j) = 1;
                end    
            end
            accuracy = mean(double(predictions == LR.y)) * 100;
            fprintf('Train Accuracy: %f \n', accuracy);
        end
    end
end