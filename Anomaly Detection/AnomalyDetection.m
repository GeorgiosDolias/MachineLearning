classdef AnomalyDetection
    properties
        X = [];
        Xval = [];
        yval = [];
    end
    methods
        %Constructor
        function AD = AnomalyDetection(string)
            data = load(string);
            AD.X = data.X;
            AD.Xval = data.Xval;           
            AD.yval = data.yval;
            if size(AD.Xval,1) ~= size(AD.yval,1) 
               error('Invalid imported data'); 
            end            
        end
        
        
        %   This function estimates the parameters of a 
        %   Gaussian distribution using the data in X. 
        %   The input X is the dataset with each n-dimensional data point in one row
        %   The output is an n-dimensional vector mu, the mean of the data set
        %   and the variances sigma^2, an n x 1 vector
        % 
        function [mu sigma2] = EstimateGaussian(AD,X)
        
            % Useful variables
            [m, n] = size(X);

            % You should return these values correctly
            mu = zeros(n, 1);
            sigma2 = zeros(n, 1);

            mu = sum(X,1)/m;

            sigma2 = sum((X-mu).^2,1)/m;
        
        end
        
        %    Computes the probability density function of the
        %    multivariate gaussian distribution.
        %    Computes the probability 
        %    density function of the examples X under the multivariate gaussian 
        %    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
        %    treated as the covariance matrix. If Sigma2 is a vector, it is treated
        %    as the \sigma^2 values of the variances in each dimension (a diagonal
        %    covariance matrix) 
        function p = MultivariateGaussian(AD,X, mu, Sigma2)

            k = length(mu);

            if (size(Sigma2, 2) == 1) || (size(Sigma2, 1) == 1)
                Sigma2 = diag(Sigma2);
            end

            X = bsxfun(@minus, X, mu(:)');
            p = (2 * pi) ^ (- k / 2) * det(Sigma2) ^ (-0.5) * ...
                exp(-0.5 * sum(bsxfun(@times, X * pinv(Sigma2), X), 2));

        end
        
        
        %   Visualize the dataset and its estimated distribution.
        %   This visualization shows you the 
        %   probability density function of the Gaussian distribution. Each example
        %   has a location (x1, x2) that depends on its feature values.
        %   Find Outliers 
        function VisualizeFit(AD,X, mu, sigma2)
        

            [X1,X2] = meshgrid(0:.5:35); 
            Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
            Z = reshape(Z,size(X1));

            plot(X(:, 1), X(:, 2),'bx');
            hold on;
            % Do not plot if there are infinities
            if (sum(isinf(Z)) == 0)
                contour(X1, X2, Z, 10.^(-20:3:0)');
            end
            hold off;

        end
        
        %   Find the best threshold (epsilon) to use for selecting
        %   outliers
        %   Finds the best threshold to use for
        %   selecting outliers based on the results from a
        %   validation set (pval) and the ground truth (yval).
        function [bestEpsilon bestF1] = SelectThreshold(AD,X,yval, p, pval)
        
            bestEpsilon = 0;
            bestF1 = 0;
            F1 = 0;

            stepsize = (max(pval) - min(pval)) / 1000;
            for epsilon = min(pval):stepsize:max(pval)

                pval;
                epsilon;


                predictions = pval < epsilon;

                tp = sum((predictions == 1)& (yval == 1));
                fp = sum((predictions == 1) & (yval == 0));
                fn = sum((predictions == 0) & (yval == 1));

                prec = tp/(tp+fp);
                rec = tp/(tp+fn);

                F1 = (2*prec*rec)/(prec + rec);

                if(epsilon == 0.1004)
                    tp;
                    fp;
                    fn;
                end

                if F1 > bestF1
                    bestF1 = F1;
                    bestEpsilon = epsilon;
                end
            end
                
            %  Find the outliers in the training set and plot the
            outliers = find(p < bestEpsilon);

            %  Draw a red circle around those outliers
            hold on
            plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
            hold off

        end
    end
end        
        