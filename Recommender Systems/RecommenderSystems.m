classdef RecommenderSystems
    properties
        Y = [];
        R = [];
        %{
        X = [];
        Theta = [];
        num_users = [];
        num_movies = [];
        num_features = [];
        %}
    end
    methods
        %Constructor
        function RS = RecommenderSystems(string)
            data = load(string);            
            RS.R = data.R;   
            RS.Y = data.Y;
            %{
            RS.X = data.X;
            RS.Theta = data.Theta;
            RS.num_users = data.num_users;
            RS.num_movies = data.num_movies;
            RS.num_features = data.num_features;
            %}
            if size(RS.Y,1) ~= size(RS.R,1) 
               error('Invalid imported data'); 
            end            
        end
        
        
        %   Collaborative filtering cost function
        %   Returns the cost and gradient for the
        %   collaborative filtering problem.
        function [J, grad] = CofiCostFunc(RS,params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
        

            % Unfold the U and W matrices from params
            X = reshape(params(1:num_movies*num_features), num_movies, num_features);
            Theta = reshape(params(num_movies*num_features+1:end), ...
                            num_users, num_features);


            % You need to return the following values correctly
            J = 0;
            X_grad = zeros(size(X));
            Theta_grad = zeros(size(Theta));


            size(X);
            size(Theta');

            %   Cost function calculation 

            sum2 = 0;
            h=X*Theta';

            size(h);
            size(Y);

            %{
            for i=1:m,
                sum= sum + (h(i)-y(i))^2;
            end
            %}
            cost= (h-Y).^2;

            sum2 = sum(cost(R == 1));

            %sum2 = sum(sum(R.*cost));

            J = (1/2)*sum2;

            % Reguarized cost

            J = J + lambda* sum(sum(Theta.^2))/2 + lambda * sum(sum(X.^2))/2;


            X_grad = ((X*Theta'-Y).*R)*Theta;
            Theta_grad = ((X*Theta'-Y).*R)'*X;


            % Regularized Gradient

            X_grad = X_grad + lambda * X;
            Theta_grad = Theta_grad + lambda *Theta;
            % =============================================================

            grad = [X_grad(:); Theta_grad(:)];

        end
        
        
        %   Creates a collaborative filering problem 
        %   to check your cost function and gradients
        %   Creates a collaborative filering problem 
        %   to check your cost function and gradients, it will output the 
        %   analytical gradients produced by your code and the numerical gradients 
        %   (computed using computeNumericalGradient). These two gradient 
        %   computations should result in very similar values.
        function CheckCostFunction(RS,lambda)
        

            % Set lambda
            if ~exist('lambda', 'var') || isempty(lambda)
                lambda = 0;
            end

            %% Create small problem
            X_t = rand(4, 3);
            Theta_t = rand(5, 3);

            % Zap out most entries
            Y = X_t * Theta_t';
            Y(rand(size(Y)) > 0.5) = 0;
            R = zeros(size(Y));
            R(Y ~= 0) = 1;

            %% Run Gradient Checking
            X = randn(size(X_t));
            Theta = randn(size(Theta_t));
            num_users = size(Y, 2);
            num_movies = size(Y, 1);
            num_features = size(Theta_t, 2);

            numgrad = ComputeNumericalGradient(RS, ...
                            @(t) cofiCostFunc(t, Y, R, num_users, num_movies, ...
                                            num_features, lambda), [X(:); Theta(:)]);

            [cost, grad] = CofiCostFunc(RS,[X(:); Theta(:)],  Y, R, num_users, ...
                                      num_movies, num_features, lambda);

            disp([numgrad grad]);
            fprintf(['The above two columns you get should be very similar.\n' ...
                     '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);

                diff = norm(numgrad-grad)/norm(numgrad+grad);
                fprintf(['If your cost function implementation is correct, then \n' ...
                         'the relative difference will be small (less than 1e-9). \n' ...
                         '\nRelative Difference: %g\n'], diff);

        end
        
         
        %   Computes the gradient using "finite differences"
        %   and gives us a numerical estimate of the gradient.
        %   Computes the numerical gradient of the
        %   function J around theta. Calling y = J(theta) should
        %   return the function value at theta.
        function numgrad = ComputeNumericalGradient(RS,J, theta)
        
            % Notes: The following code implements numerical gradient checking, and 
            %        returns the numerical gradient.It sets numgrad(i) to (a numerical 
            %        approximation of) the partial derivative of J with respect to the 
            %        i-th input argument, evaluated at theta. (i.e., numgrad(i) should 
            %        be the (approximately) the partial derivative of J with respect 
            %        to theta(i).)
            %                

            numgrad = zeros(size(theta));
            perturb = zeros(size(theta));
            e = 1e-4;
            for p = 1:numel(theta)
                % Set perturbation vector
                perturb(p) = e;
                loss1 = J(theta - perturb);
                loss2 = J(theta + perturb);
                % Compute Numerical Gradient
                numgrad(p) = (loss2 - loss1) / (2*e);
                perturb(p) = 0;
            end

        end

        
        %   Reads the fixed movie list in movie.txt and returns a
        %   cell array of the words.
        %   Reads the fixed movie list in movie.txt 
        %   and returns a cell array of the words in movieList.
        function MovieList = LoadMovieList(RS)
        


            %% Read the fixed movieulary list
            fid = fopen('movie_ids.txt');

            % Store all movies in cell array movie{}
            n = 1682;  % Total number of movies 

            MovieList = cell(n, 1);
            for i = 1:n
                % Read line
                line = fgets(fid);
                % Word Index (can ignore since it will be = i)
                [idx, movieName] = strtok(line, ' ');
                % Actual Word
                    MovieList{i} = strtrim(movieName);
                end
                fclose(fid);

        end
        
        %   Preprocess data by subtracting mean rating for every 
        %   movie (every row)
        %   Normalized Y so that each movie
        %   has a rating of 0 on average, and returns the mean rating in Ymean.
        function [Ynorm, Ymean] = NormalizeRatings(RS,Y, R)
        

            [m, n] = size(Y);
            Ymean = zeros(m, 1);
            Ynorm = zeros(size(Y));
            for i = 1:m
                idx = find(R(i, :) == 1);
                Ymean(i) = mean(Y(i, idx));
                Ynorm(i, idx) = Y(i, idx) - Ymean(i);
            end

        end
        
        
        %  Trains a collaborative filtering model on a movie rating 
        %  dataset
        function [X,Theta,Ymean] = LearningMovieRatings(RS,my_ratings,string)
            
            fprintf('\nTraining collaborative filtering...\n');

            %  Load data
            load(string);

            %  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
            %  943 users
            %
            %  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
            %  rating to movie i

            %  Add our own ratings to the data matrix
            Y = [my_ratings Y];
            R = [(my_ratings ~= 0) R];

            %  Normalize Ratings
            [Ynorm, Ymean] = normalizeRatings(Y, R);

            %  Useful Values
            num_users = size(Y, 2);
            num_movies = size(Y, 1);
            num_features = 10;

            % Set Initial Parameters (Theta, X)
            X = randn(num_movies, num_features);
            Theta = randn(num_users, num_features);

            initial_parameters = [X(:); Theta(:)];

            % Set options for fmincg
            options = optimset('GradObj', 'on', 'MaxIter', 100);

            % Set Regularization
            lambda = 10;
            theta = fmincg (@(t)(CofiCostFunc(RS,t, Ynorm, R, num_users, num_movies, ...
                                            num_features, lambda)), ...
                            initial_parameters, options);

            % Unfold the returned theta back into U and W
            X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
            Theta = reshape(theta(num_movies*num_features+1:end), ...
                            num_users, num_features);

            fprintf('Recommender system learning completed.\n');
        end
        
        
        
        %  After training the model, you can now make recommendations by computing
        %  the predictions matrix.

        function Recommendations(RS,X,Theta,Ymean,my_ratings)
            
            p = X * Theta';
            my_predictions = p(:,1) + Ymean;

            movieList = LoadMovieList(RS);

            [r, ix] = sort(my_predictions, 'descend');
            fprintf('\nTop recommendations for you:\n');
            for i=1:10
                j = ix(i);
                fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
                        movieList{j});
            end

            fprintf('\n\nOriginal ratings provided:\n');
            for i = 1:length(my_ratings)
                if my_ratings(i) > 0 
                    fprintf('Rated %d for %s\n', my_ratings(i), ...
                             movieList{i});
                end
            end
        end
        
        
    end
end