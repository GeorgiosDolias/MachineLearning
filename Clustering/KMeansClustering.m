classdef KMeansClustering
    properties
        X = [];
        data = [];
    end
    methods
        %Constructor
        function KMC = KMeansClustering(string)
            data = load(string);
            KMC.X = data.X;
            for i= 2:size(KMC.X,2)               
               if size(KMC.X(1,:),1) ~= size(KMC.X(i,:),1) 
                  error('Invalid imported data'); 
               end
            end
        end
        
        %   Computes the centroid memberships for every example.
        %   Returns the closest centroids in idx for a dataset X
        %    where each row is a single example. idx = m x 1 
        %   vector of centroid assignments (i.e. each entry in range [1..K])
        
        
        function idx = FindClosestCentroids(KMC,X, centroids)            

            % Set K
            K = size(centroids, 1);

            % You need to return the following variables correctly.
            idx = zeros(size(X,1), 1);

            for i = 1:length(idx)   % Loop throuth every training example
                min = 0;
                %X(i,:)-centroids(j,1)

                for j =1:K          % Loop through every centroid    
                    distance = X(i,:)-centroids(j,:);
                    distance = sum(distance.^2);
                    size(distance);
                    %^2 + (X(i,2)-centroids(j,2))^2);
                    %centroids(1,j)
                    %i;
                    if(distance < min || j==1 )
                        min = distance;
                       idx(i)= j;
                    end
                    if i == 1
                        distance;
                    end
                end
            end
            size(idx);
        end
        
        
        %   Returns the new centroids by computing the means of the 
        %   data points assigned to each centroid.
        %   returns the new centroids by computing the means of the
        %   data points assigned to each centroid. It is
        %   given a dataset X where each row is a single data point, a vector
        %   idx of centroid assignments (i.e. each entry in range [1..K]) for each
        %   example, and K, the number of centroids. You should return a matrix
        %   centroids, where each row of centroids is the mean of the data points
        %   assigned to it.
        function centroids = ComputeCentroids(KMC,X, idx, K)            

            % Useful variables
            [m n] = size(X);

            % You need to return the following variables correctly.
            centroids = zeros(K, n);
            
            size(X(idx(:)==3,:))

            for j=1:K       % Loop over all centroids
                sum1 = 0;
               
                for i=1:m
                    if idx(i)==j
                        sum1 = sum1 + X(i,:);
                    end
                end
                centroids(j,:) = sum1./nnz(idx==j);     % Sum of X values connected with centroid j divided by
            end                                         % the number of occurence of centroid j in idx
        end
        
        
        %   Runs the K-Means algorithm on data matrix X, where each row of X
        %   is a single example
        %   Runs the K-Means algorithm on data matrix X, where each 
        %   row of X is a single example. It uses initial_centroids used as the
        %   initial centroids. max_iters specifies the total number of interactions 
        %   of K-Means to execute. plot_progress is a true/false flag that 
        %   indicates if the function should also plot its progress as the 
        %   learning happens. This is set to false by default. runkMeans returns 
        %   centroids, a Kxn matrix of the computed centroids and idx, a m x 1 
        %   vector of centroid assignments (i.e. each entry in range [1..K]
        function [centroids, idx] = RunkMeans(KMC,X, initial_centroids, ...
                                      max_iters, plot_progress)            

            % Set default value for plot progress
            if ~exist('plot_progress', 'var') || isempty(plot_progress)
                plot_progress = false;
            end

            % Plot the data if we are plotting progress
            if plot_progress
                figure;
                hold on;
            end

            % Initialize values
            [m n] = size(X);
            K = size(initial_centroids, 1);
            centroids = initial_centroids;
            previous_centroids = centroids;
            idx = zeros(m, 1);

            % Run K-Means
            for i=1:max_iters

                % Output progress
                fprintf('K-Means iteration %d/%d...\n', i, max_iters);
                if exist('OCTAVE_VERSION')
                    fflush(stdout);
                end

                % For each example in X, assign it to the closest centroid
                idx = FindClosestCentroids(KMC,X, centroids);

                % Optionally, plot progress here
                if plot_progress
                    plotProgresskMeans(X, centroids, previous_centroids, idx, K, i);
                    previous_centroids = centroids;
                    fprintf('Press enter to continue.\n');
                    pause;
                end

                % Given the memberships, compute new centroids
                centroids = ComputeCentroids(KMC,X, idx, K);
            end

            % Hold off if we are plotting progress
            if plot_progress
                hold off;
            end

        end
        
        
        %   This function initializes K centroids that are to be 
        %   used in K-Means on the dataset X.
        %   Returns K initial centroids to be
        %   used with the K-Means on the dataset X
        function centroids = kMeansInitCentroids(KMC,X, K)

            % You should return this values correctly
            centroids = zeros(K, size(X, 2));

            % Randomly reorder the indices of examples
            randidx = randperm(size(X,1));

            % Take the first K examples as centroid
            centroids = X(randidx(1:K),:);
        end
        
        
        %  Uses K-Means to compress an image.
        %  First run K-Means on the colors of the pixels in the image and
        %  then you will map each pixel onto its closest centroid.        
        function [X,A,centroids, idx] = KMConPixels(KMC, string, K, max_iters)
        
            %  Load an image of a bird
            A = double(imread(string));

            % If imread does not work for you, you can try instead
            %   load ('bird_small.mat');

            A = A / 255; % Divide by 255 so that all values are in the range 0 - 1

            % Size of the image
            img_size = size(A);

            % Reshape the image into an Nx3 matrix where N = number of pixels.
            % Each row will contain the Red, Green and Blue pixel values
            % This gives us our dataset matrix X that we will use K-Means on.
            X = reshape(A, img_size(1) * img_size(2), 3);

            % When using K-Means, it is important the initialize the centroids
            % randomly. 
            % You should complete the code in kMeansInitCentroids.m before proceeding
            initial_centroids = kMeansInitCentroids(KMC,X, K);

            % Run K-Means
            [centroids, idx] = RunkMeans(KMC,X, initial_centroids, max_iters);

        end
        
        
         %  Uses the clusters of K-Means to compress an image.
         %  To do this, we first find the closest clusters for
         %  each example.
        function ImageCompression(KMC,X,A,K,centroids)
           

            fprintf('\nApplying K-Means to compress an image.\n\n');

            % Find closest cluster members
            idx = FindClosestCentroids(KMC,X, centroids);

            % Essentially, now we have represented the image X as in terms of the
            % indices in idx. 

            % We can now recover the image from the indices (idx) by mapping each pixel
            % (specified by its index in idx) to the centroid value
            X_recovered = centroids(idx,:);
            
            % Size of the image
            img_size = size(A);
            
            % Reshape the recovered image into proper dimensions
            X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

            % Display the original image 
            subplot(1, 2, 1);
            imagesc(A); 
            title('Original');

            % Display compressed image side by side
            subplot(1, 2, 2);
            imagesc(X_recovered)
            title(sprintf('Compressed, with %d colors.', K));


            fprintf('Program paused. Press enter to continue.\n');
            pause;
        
        end
    end
end