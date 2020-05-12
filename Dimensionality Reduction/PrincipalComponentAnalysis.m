classdef PrincipalComponentAnalysis
    properties
        X = [];
    end
    methods
        %Constructor
        function PCA = PrincipalComponentAnalysis(string)
            data = load(string);
            PCA.X = data.X;
            for i= 2:size(PCA.X,2)               
               if size(PCA.X(1,:),1) ~= size(PCA.X(i,:),1) 
                  error('Invalid imported data'); 
               end
            end
        end
        
        
        %   Normalizes the features in X .
        %   Returns a normalized version of X where
        %   the mean value of each feature is 0 and the standard deviation
        %   is 1. This is often a good preprocessing step to do when
        %   working with learning algorithms.
        function [X_norm, mu, sigma] = FeatureNormalize(PCA,X)
            
            mu = mean(X);
            X_norm = bsxfun(@minus, X, mu);

            sigma = std(X_norm);
            X_norm = bsxfun(@rdivide, X_norm, sigma);            
            
        end     
        
        
        %   Runs principal component analysis on the dataset X.
        %   Computes eigenvectors of the covariance matrix of X
        %   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
        function [U, S] = pca(PCA,X)
        
            % Useful values
            [m, n] = size(X);

            % You need to return the following variables correctly.
            U = zeros(n);
            S = zeros(n);

            S = (X'*X)/m;
            
            %mu = mean(PCA.X);
            
            %   Computes the eigenvectors and eigenvalues of the
            %   covariance matrix.            
            [U, S, V] = svd(S);

        end
        

        %  Implements the projection step to map the data onto the 
        %  first k eigenvectors. The code will then plot the data in this reduced 
        %  dimensional space. This will show you what the data looks like when 
        %  using only the corresponding eigenvectors to reconstruct it.
        
        %   Project the data onto K dimension.
        %   Computes the reduced data representation when projecting only 
        %   on to the top k eigenvectors.
        %   Computes the projection of the normalized inputs X into 
        %   the reduced dimensional space spanned by the 
        %   first K columns of U. It returns the projected examples in Z.
        function Z = ProjectData(PCA,X, U, K)
       
            % You need to return the following variables correctly.
            Z = zeros(size(X, 1), K);

            size(X);
            size(U);

            x = X';

            U_reduce = U(:,1: K);
            size(U_reduce);

            Z =  x'* U_reduce;
            size(Z);

        end
        
        
        %   Recovers an approximation of the original data when using the 
        %   projected data.
        %   Recovers an approximation of the original data
        %   that has been reduced to K dimensions. It returns the
        %   approximate reconstruction in X_rec.        %
        function X_rec = RecoverData(PCA,Z, U, K)
        
            X_rec = zeros(size(Z, 1), size(U, 1));

            size(Z)

            size(U(:,1: K)')
            U_reduce = U(:,1: K)';

             v = Z';

            X_rec = v' * U_reduce;

        end
        
        
        %   Displays 2D data stored in X in a nice grid.
        %   It returns the figure handle h and the 
        %   displayed array if requested.
        function [h, display_array] = DisplayData(PCA,X, example_width)
        

            % Set example_width automatically if not passed in
            if ~exist('example_width', 'var') || isempty(example_width) 
                example_width = round(sqrt(size(X, 2)));
            end

            % Gray Image
            colormap(gray);

            % Compute rows, cols
            [m n] = size(X);
            example_height = (n / example_width);

            % Compute number of items to display
            display_rows = floor(sqrt(m));
            display_cols = ceil(m / display_rows);

            % Between images padding
            pad = 1;

            % Setup blank display
            display_array = - ones(pad + display_rows * (example_height + pad), ...
                                   pad + display_cols * (example_width + pad));

            % Copy each example into a patch on the display array
            curr_ex = 1;
            for j = 1:display_rows
                for i = 1:display_cols
                    if curr_ex > m, 
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
                if curr_ex > m, 
                    break; 
                end
            end

            % Display Image
            h = imagesc(display_array, [-1 1]);

            % Do not show axis
            axis image off

            drawnow;
        end
    end
end