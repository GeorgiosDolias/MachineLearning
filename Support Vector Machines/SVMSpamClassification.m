classdef SVMSpamClassification
    properties
        X = [];
        x=[];
        y = [];
        Xtest = [];
        ytest = [];
        data = [];
        m = 1;
        iterations = 1000        
        theta = []
    end
    methods
        %Constructor
        function SVMSC = SVMSpamClassification(string)
            
            structure = load(string);
       
            SVMSC.X = structure.X;
            %SVMSC.x = structure.x;
            SVMSC.y = structure.y;
            %.Xtest = structure.Xtest;
            %SVMSC.ytest = structure.ytest;
            SVMSC.m = size(SVMSC.X, 1);
            if size(SVMSC.X,1) ~= size(SVMSC.y,1)
                    error('Invalid imported data');
            end
            
        end
        
        
        %  To use an SVM to classify emails into Spam v.s. Non-Spam, 
        %  you first need to convert each email into a vector of features. 
        
        %   Reads a file and returns its entire contents in file_contents
        function file_contents = ReadFile(SVMSC,filename)
            
            % Load File
            fid = fopen(filename);
            if fid
                file_contents = fscanf(fid, '%c', inf);
                fclose(fid);
            else
                file_contents = '';
                fprintf('Unable to open %s\n', filename);
            end
        end
        
        
        %   preprocesses the body of an email and returns a list of
        %   indices of the words contained in the email.  
        function word_indices = ProcessEmail(SVMSC,email_contents)
           

            % Load Vocabulary
            vocabList = getVocabList();

            % Init return value
            word_indices = [];

            % ========================== Preprocess Email ===========================

            % Find the Headers ( \n\n and remove )
            % Uncomment the following lines if you are working with raw emails with the
            % full headers

            % hdrstart = strfind(email_contents, ([char(10) char(10)]));
            % email_contents = email_contents(hdrstart(1):end);

            % Lower case
            email_contents = lower(email_contents);

            % Strip all HTML
            % Looks for any expression that starts with < and ends with > and replace
            % and does not have any < or > in the tag it with a space
            email_contents = regexprep(email_contents, '<[^<>]+>', ' ');

            % Handle Numbers
            % Look for one or more characters between 0-9
            email_contents = regexprep(email_contents, '[0-9]+', 'number');

            % Handle URLS
            % Look for strings starting with http:// or https://
            email_contents = regexprep(email_contents, ...
                                       '(http|https)://[^\s]*', 'httpaddr');

            % Handle Email Addresses
            % Look for strings with @ in the middle
            email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

            % Handle $ sign
            email_contents = regexprep(email_contents, '[$]+', 'dollar');


            % ========================== Tokenize Email ===========================

            % Output the email to screen as well
            fprintf('\n==== Processed Email ====\n\n');

            % Process file
            l = 0;

            while ~isempty(email_contents)

                % Tokenize and also get rid of any punctuation
                [str, email_contents] = ...
                   strtok(email_contents, ...
                          [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);

                % Remove any non alphanumeric characters
                str = regexprep(str, '[^a-zA-Z0-9]', '');

                % Stem the word 
                % (the porterStemmer sometimes has issues, so we use a try catch block)
                try str = porterStemmer(strtrim(str)); 
                catch str = ''; continue;
                end

                % Skip the word if it is too short
                if length(str) < 1
                   continue;
                end

            

                for i=1:length(vocabList)
                    if(strcmp(str,vocabList{i}))
                        word_indices = [word_indices ; i];
                        break;
                    end
                end

                % Print to screen, ensuring that the output lines are not too long
                if (l + length(str) + 1) > 78
                    fprintf('\n');
                    l = 0;
                end
                fprintf('%s ', str);
                l = l + length(str) + 1;

            end

            % Print footer
            fprintf('\n\n=========================\n');

        end
        
        %   Converts each email into a vector of features in R^n. 
        %   Takes in a word_indices vector and produces a feature vector
        %   from the word indices.
        %   Takes in a word_indices vector and 
        %   produces a feature vector from the word indices. 
        
        function x = EmailFeatures(SVMSC,word_indices)
            

            % Total number of words in the dictionary
            n = 1899;

            % You need to return the following variables correctly.
            x = zeros(n, 1);

            for i =1:n
                for j = 1:length(word_indices)
                    if(i==word_indices(j))
                        x(i) = 1;
                        break;
                    end
                end
            end

        end

        %   Trains a linear classifier to determine if an
        %   email is Spam or Not-Spam.
        %   Trains an SVM classifier using a simplified version of the SMO 
        %   algorithm. It trains an SVM classifier and returns trained
        %   model. X is the matrix of training examples. 
        %   Each row is a training example, and the jth column holds the
        %   jth feature. Y is a column matrix containing 1 for positive examples 
        %   and 0 for negative examples.  C is the standard SVM regularization 
        %   parameter. tol is a tolerance value used for determining equality of 
        %   floating point numbers. max_passes controls the number of iterations
        %   over the dataset (without changes to alpha) before the algorithm quits.
       
        function [model] = SVMTrain(SVMSC,X, Y, C, kernelFunction, ...
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
        
        %   Returns a vector of predictions using a trained SVM model
        %   (svmTrain). 
        %   X is a mxn matrix where there each 
        %   example is a row. model is a svm model returned from svmTrain.
        %   predictions pred is a m x 1 column of predictions of {0, 1} values.
        function pred = SVMPredict(SVMSC,model, X,y)
            

            % Check if we are getting a column vector, if so, then assume that we only
            % need to do prediction for a single example
            if (size(X, 2) == 1)
                % Examples should be in rows
                X = X';
            end

            % Dataset 
            m = size(X, 1);
            p = zeros(m, 1);
            pred = zeros(m, 1);

            if strcmp(func2str(model.kernelFunction), 'linearKernel')
                % We can use the weights and bias directly if working with the 
                % linear kernel
                p = X * model.w + model.b;
            elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
                % Vectorized RBF Kernel
                % This is equivalent to computing the kernel on every pair of examples
                X1 = sum(X.^2, 2);
                X2 = sum(model.X.^2, 2)';
                K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
                K = model.kernelFunction(1, 0) .^ K;
                K = bsxfun(@times, model.y', K);
                K = bsxfun(@times, model.alphas', K);
                p = sum(K, 2);
            else
                % Other Non-linear kernel
                for i = 1:m
                    prediction = 0;
                    for j = 1:size(model.X, 1)
                        prediction = prediction + ...
                            model.alphas(j) * model.y(j) * ...
                            model.kernelFunction(X(i,:)', model.X(j,:)');
                    end
                    p(i) = prediction + model.b;
                end
            end

            % Convert predictions into 0 / 1
            pred(p >= 0) =  1;
            pred(p <  0) =  0;
            
            fprintf('Training Accuracy: %f\n', mean(double(pred == y)) * 100);
            
        end

        
        %  Since the model we are training is a linear SVM, we can inspect the
        %  weights learned by the model to understand better how it is determining
        %  whether an email is spam or not. The following code finds the words with
        %  the highest weights in the classifier. Informally, the classifier
        %  'thinks' that these words are the most likely indicators of spam.

        %   Reads the fixed vocabulary list in vocab.txt and returns a
        %   cell array of the words in vocabList.

        function TopPredictors(SVMSC,model)
            
            
            % Sort the weights and obtain the vocabulary list
            [weight, idx] = sort(model.w, 'descend');
            
            %% Read the fixed vocabulary list
            fid = fopen('vocab.txt');

            % Store all dictionary words in cell array vocab{}
            n = 1899;  % Total number of words in the dictionary

            % For ease of implementation, we use a struct to map the strings => integers
            % In practice, you'll want to use some form of hashmap
            vocabList = cell(n, 1);
            for i = 1:n
                % Word Index (can ignore since it will be = i)
                fscanf(fid, '%d', 1);
                % Actual Word
                vocabList{i} = fscanf(fid, '%s', 1);
            end
            fclose(fid);
            
          

            fprintf('\nTop predictors of spam: \n');
            for i = 1:15
                fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
            end
            
        end

        
        %  Now that you've trained the spam classifier, you can use it on your own
        %  emails! 
        %  The following code reads in one of these emails and then uses your 
        %  learned SVM classifier to determine whether the email is Spam or 
        %  Not Spam
        function CheckEmail(SVMSC,filename,model)
            
            % Read and predict
            file_contents = ReadFile(SVMSC,filename);
            word_indices  = ProcessEmail(SVMSC,file_contents);
            X             = EmailFeatures(SVMSC,word_indices);
            
            % Check if we are getting a column vector, if so, then assume that we only
            % need to do prediction for a single example
            if (size(X, 2) == 1)
                % Examples should be in rows
                X = X';
            end

            % Dataset 
            m = size(X, 1);
            p = zeros(m, 1);
            pred = zeros(m, 1);

            if strcmp(func2str(model.kernelFunction), 'linearKernel')
                % We can use the weights and bias directly if working with the 
                % linear kernel
                p = X * model.w + model.b;
            elseif strfind(func2str(model.kernelFunction), 'gaussianKernel')
                % Vectorized RBF Kernel
                % This is equivalent to computing the kernel on every pair of examples
                X1 = sum(X.^2, 2);
                X2 = sum(model.X.^2, 2)';
                K = bsxfun(@plus, X1, bsxfun(@plus, X2, - 2 * X * model.X'));
                K = model.kernelFunction(1, 0) .^ K;
                K = bsxfun(@times, model.y', K);
                K = bsxfun(@times, model.alphas', K);
                p = sum(K, 2);
            else
                % Other Non-linear kernel
                for i = 1:m
                    prediction = 0;
                    for j = 1:size(model.X, 1)
                        prediction = prediction + ...
                            model.alphas(j) * model.y(j) * ...
                            model.kernelFunction(X(i,:)', model.X(j,:)');
                    end
                    p(i) = prediction + model.b;
                end
            end

            % Convert predictions into 0 / 1
            pred(p >= 0) =  1;
            pred(p <  0) =  0;
            

            fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, pred);
            fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
        end