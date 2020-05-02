classdef OneVariableLinearRegression
    properties
        X = []
        y = []
        data = []
        m = 1
        iterations = 1500
        alpha = 0.01; %     Learning rate
        theta = zeros(2, 1)
    end
    methods
        %Constructor
        function LR = OneVariableLinearRegression(string)
            LR.data = load(string);
            if size(LR.data,2) == 2
                LR.X = LR.data(:, 1);
                LR.y = LR.data(:, 2);
                LR.m = length(LR.y);
                if size(LR.X,1) ~= size(LR.y,1)
                    error('Invalid imported data');
                end
            else
                error('Invalid imported data');
            end
        end
         
        %   PLOTDATA(x,y) plots the data points and gives the figure 
        %   axes labels of population and profit.
        function PlotData(LR)
            figure;
            plot(LR.X,LR.y,'rx','MarkerSize',10);
            ylabel('Profits in $10,000s');
            xlabel('Population of City in 10,000s');
        end
        %   Computes the cost of using theta as the parameter
        %   for linear regression to fit the data points in X and y
        function J = ComputeCost(LR)
            J = 0;
            sum = 0;
            LR.X = [ones(LR.m, 1), LR.data(:,1)]; 
            h=LR.X*LR.theta;

            for i=1:LR.m
                sum= sum + (h(i)-LR.y(i))^2;
            end
            
            J = (1/(2*LR.m))*sum;
        end
        %   Performs gradient descent to learn theta. It updates theta
        %   by taking iterations gradient steps with learning rate alpha    
        function Theta = GradientDescent(LR)
            LR.X = [ones(LR.m, 1), LR.data(:,1)];
            
            for iter = 1:LR.iterations

                sum1 = 0;
                sum2 = 0;

                h=LR.X*LR.theta;

                for i=1:LR.m
                    sum1 = sum1 + (h(i)-LR.y(i))*LR.X(i,1);
                    sum2 = sum2 + (h(i)-LR.y(i))*LR.X(i,2);
                end

                temp1 = LR.theta(1) - LR.alpha*(1/LR.m)*sum1;
                temp2 = LR.theta(2) - LR.alpha*(1/LR.m)*sum2;
                Theta(1) =temp1;
                Theta(2) =temp2;

                LR.theta(1) = Theta(1);
                LR.theta(2) = Theta(2);
            end
        end
        %Plot linear fit
        function PlotLinearFit(LR)
            
            hold on; % keep previous plot visible
            LR.X = [ones(LR.m, 1), LR.data(:,1)];
            plot(LR.X(:,2), LR.X*LR.GradientDescent', '-')
            legend('Training data', 'Linear regression')
            hold off % don't overlay any more plots on this figure
        end
        % Predict values for various population sizes
        function Predictions(LR, example)
            predict = example *LR.GradientDescent';
            fprintf('For population = %5.2f , we predict a profit of %f\n',...
                    example(2)*10000,predict*10000);
        end
        % Plot Cost function for different values of theta0,theta1
        function PlotCost(LR)
            fprintf('Visualizing J(theta_0, theta_1) ...\n')

            % Grid over which we will calculate J
            theta0_vals = linspace(-10, 10, 100);
            theta1_vals = linspace(-1, 4, 100);

            % initialize J_vals to a matrix of 0's
            J_vals = zeros(length(theta0_vals), length(theta1_vals));

            % Fill out J_vals
            for i = 1:length(theta0_vals)
                for j = 1:length(theta1_vals)
                  t = [theta0_vals(i); theta1_vals(j)];
                  LR.theta = t;
                  J_vals(i,j) = LR.ComputeCost;
                end
            end

            % Because of the way meshgrids work in the surf command, 
            % we need to transpose J_vals before calling surf, or else
            % the axes will be flipped
            J_vals = J_vals';
            % Surface plot
            figure;
            surf(theta0_vals, theta1_vals, J_vals)
            xlabel('\theta_0'); ylabel('\theta_1');
            
            % Contour plot
            figure;
            % Plot J_vals as 15 contours spaced logarithmically 
            %between 0.01 and 100
            contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
            xlabel('\theta_0'); ylabel('\theta_1');
            hold on;
            th = LR.GradientDescent;
            plot(th(1), th(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
        end
    end
end