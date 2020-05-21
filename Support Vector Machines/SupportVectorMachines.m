classdef SupportVectorMachines
    properties
        X = [];
        y = [];
        Xval = [];
        yval = [];
        m = 1;
    end
    methods
        %Constructor
        function SVM = SupportVectorMachines(string)
            if string ~= ""
                structure = load(string);

                SVM.X = structure.X;
                SVM.y = structure.y;
                SVM.m = size(SVM.X, 1);
                if size(SVM.X,1) ~= size(SVM.y,1)
                        error('Invalid imported data');
                end
            end
        end
        
        %   Plots the data points X and y into a new figure. 
        %   Plots the data points with + for the positive examples
        %   and o for the negative examples. X is assumed to be a Mx2 matrix.
        function PlotData(SVM,X,y)
            %Create New Figure
            figure; hold on;

            % Find Indices of Positive and Negative Examples
            pos = find(y == 1); neg = find(y == 0);

            % Plot Examples
            plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
            %hold on;
            plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'c', 'MarkerSize', 7);
            hold off;
        end
        
        %   Trains an SVM classifier using a simplified version of the SMO 
        %   algorithm. It trains an SVM classifier and returns trained
        %   model. X is the matrix of training examples. 
        %   Each row is a training example, and the jth column holds the
        %   jth feature. Y is a column matrix containing 1 for positive examples 
        %   and 0 for negative examples.  C is the standard SVM regularization 
        %   parameter. tol is a tolerance value used for determining equality of 
        %   floating point numbers. max_passes controls the number of iterations
        %   over the dataset (without changes to alpha) before the algorithm quits.
       
        function [model] = SVMTrain(SVM,X, Y, C, kernelFunction, ...
                            tol, max_passes)
            
            if ~exist('tol', 'var') || isempty(tol)
                tol = 1e-3;
            end

            if ~exist('max_passes', 'var') || isempty(max_passes)
                max_passes = 5;
            end

            % Data parameters
            m = size(X, 1);
            n = size(X, 2);

            % Map 0 to -1
            Y(Y==0) = -1;

            % Variables
            alphas = zeros(m, 1);
            b = 0;
            E = zeros(m, 1);
            passes = 0;
            eta = 0;
            L = 0;
            H = 0;

            % Pre-compute the Kernel Matrix since our dataset is small
            % (in practice, optimized SVM packages that handle large datasets
            %  gracefully will _not_ do this)
            % 
            % We have implemented optimized vectorized version of the Kernels here so
            % that the svm training will run faster.
            if strcmp(func2str(kernelFunction), 'linearKernel')
                % Vectorized computation for the Linear Kernel
                % This is equivalent to computing the kernel on every pair of examples
                K = X*X';
            elseif strfind(func2str(kernelFunction), 'gaussianKernel')
                % Vectorized RBF Kernel
                % This is equivalent to computing the kernel on every pair of examples
                X2 = sum(X.^2, 2);
                K = bsxfun(@plus, X2, bsxfun(@plus, X2', - 2 * (X * X')));
                K = kernelFunction(1, 0) .^ K;
            else
                % Pre-compute the Kernel Matrix
                % The following can be slow due to the lack of vectorization
                K = zeros(m);
                for i = 1:m
                    for j = i:m
                         K(i,j) = kernelFunction(X(i,:)', X(j,:)');
                         K(j,i) = K(i,j); %the matrix is symmetric
                    end
                end
            end

            % Train
            fprintf('\nTraining ...');
            dots = 12;
            while passes < max_passes

                num_changed_alphas = 0;
                for i = 1:m

                    % Calculate Ei = f(x(i)) - y(i) using (2). 
                    % E(i) = b + sum (X(i, :) * (repmat(alphas.*Y,1,n).*X)') - Y(i);
                    E(i) = b + sum (alphas.*Y.*K(:,i)) - Y(i);

                    if ((Y(i)*E(i) < -tol && alphas(i) < C) || (Y(i)*E(i) > tol && alphas(i) > 0))

                        % In practice, there are many heuristics one can use to select
                        % the i and j. In this simplified code, we select them randomly.
                        j = ceil(m * rand());
                        while j == i  % Make sure i \neq j
                            j = ceil(m * rand());
                        end

                        % Calculate Ej = f(x(j)) - y(j) using (2).
                        E(j) = b + sum (alphas.*Y.*K(:,j)) - Y(j);

                        % Save old alphas
                        alpha_i_old = alphas(i);
                        alpha_j_old = alphas(j);

                        % Compute L and H by (10) or (11). 
                        if (Y(i) == Y(j))
                            L = max(0, alphas(j) + alphas(i) - C);
                            H = min(C, alphas(j) + alphas(i));
                        else
                            L = max(0, alphas(j) - alphas(i));
                            H = min(C, C + alphas(j) - alphas(i));
                        end

                        if (L == H)
                            % continue to next i. 
                            continue;
                        end

                        % Compute eta by (14).
                        eta = 2 * K(i,j) - K(i,i) - K(j,j);
                        if (eta >= 0)
                            % continue to next i. 
                            continue;
                        end

                        % Compute and clip new value for alpha j using (12) and (15).
                        alphas(j) = alphas(j) - (Y(j) * (E(i) - E(j))) / eta;

                        % Clip
                        alphas(j) = min (H, alphas(j));
                        alphas(j) = max (L, alphas(j));

                        % Check if change in alpha is significant
                        if (abs(alphas(j) - alpha_j_old) < tol)
                            % continue to next i. 
                            % replace anyway
                            alphas(j) = alpha_j_old;
                            continue;
                        end

                        % Determine value for alpha i using (16). 
                        alphas(i) = alphas(i) + Y(i)*Y(j)*(alpha_j_old - alphas(j));

                        % Compute b1 and b2 using (17) and (18) respectively. 
                        b1 = b - E(i) ...
                             - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                             - Y(j) * (alphas(j) - alpha_j_old) *  K(i,j)';
                        b2 = b - E(j) ...
                             - Y(i) * (alphas(i) - alpha_i_old) *  K(i,j)' ...
                             - Y(j) * (alphas(j) - alpha_j_old) *  K(j,j)';

                        % Compute b by (19). 
                        if (0 < alphas(i) && alphas(i) < C)
                            b = b1;
                        elseif (0 < alphas(j) && alphas(j) < C)
                            b = b2;
                        else
                            b = (b1+b2)/2;
                        end

                        num_changed_alphas = num_changed_alphas + 1;

                    end

                end

                if (num_changed_alphas == 0)
                    passes = passes + 1;
                else
                    passes = 0;
                end

                fprintf('.');
                dots = dots + 1;
                if dots > 78
                    dots = 0;
                    fprintf('\n');
                end
                if exist('OCTAVE_VERSION')
                    fflush(stdout);
                end
            end
            fprintf(' Done! \n\n');

            % Save the model
            idx = alphas > 0;
            model.X= X(idx,:);
            model.y= Y(idx);
            model.kernelFunction = kernelFunction;
            model.b= b;
            model.alphas= alphas(idx);
            model.w = ((alphas.*Y)'*X)';

        end
       
       %    Plots a linear decision boundary learned by the
       %    SVM and overlays the data on it

       function VisualizeBoundaryLinear(SVM,X, y, model)
            
            w = model.w;
            b = model.b;
            xp = linspace(min(X(:,1)), max(X(:,1)), 100);
            yp = - (w(1)*xp + b)/w(2);
            PlotData(SVM,X, y);
            hold on;
            plot(xp, yp, '-b'); 
            hold off

       end 
       
       %   Implements the Gaussian kernel to use
       %   with the SVM.
       %   It returns a radial basis function kernel between x1 and x2      
       function sim = GaussianKernel(SVM,x1, x2, sigma)
            
            % Ensure that x1 and x2 are column vectors
            x1 = x1(:); x2 = x2(:);

            sim = exp(-sum((x1-x2).^2)/(2*sigma^2));

       end 
       
       %   plots a non-linear decision boundary learned by the SVM
       %   and overlays the data on it 
       function VisualizeBoundary(SVM,X, y, model, varargin)
           
            % Plot the training data on top of the boundary
            PlotData(SVM,X, y)

            % Make classification predictions over a grid of values
            x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
            x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
            [X1, X2] = meshgrid(x1plot, x2plot);
            vals = zeros(size(X1));
            for i = 1:size(X1, 2)
               this_X = [X1(:, i), X2(:, i)];
               vals(:, i) = svmPredict(model, this_X);
            end

            % Plot the SVM boundary
            hold on
            contour(X1, X2, vals, [0.5 0.5], 'b');
            hold off;

       end
      
      %   Returns your choice of C and sigma to use for SVM with RBF kernel
      %   You should complete this function to return the optimal C and 
      %   sigma based on a cross-validation set.       
      function [C, sigma] = Dataset3Params(SVM,X, y, Xval, yval)
            

            % You need to return the following variables correctly.
            C = 0.01;
            sigma = 0.01;
            C_values = [0.01 0.03 0.1 0.3 1 3 10 30]';

            sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30]';

            error_max = 1;
            error = 0;

            for i=1:length(C_values)
                for j=1:length(sigma_values)
                    model= svmTrain(X, y, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j)));
                    predictions = svmPredict(model, Xval);
                    error = mean(double(predictions ~= yval));

                    if error < error_max
                       C = C_values(i);
                       sigma = sigma_values(j);
                       error_max = error
                    end
                end
            end
      end
    end
end