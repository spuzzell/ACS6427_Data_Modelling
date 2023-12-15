clear;
close all;
clc;

file = 'QSAR_data.mat'; % Loads the dataset into memory 
folds  = 5;             % Number of folds
iter   = 40;            % Iterations of Bayesian Optimization
epochs = 200;           % Iterations of SVM
var    = 0.9;           % Varience target for PCA

% Pre-processing
[data]                     = loadnsort(file);
data_vis(data);
[data.pca]                 = pca_data_maker(data.Xnorm,var);
[train,test]               = kfold(data.Xnorm,data.Y,folds);
[train_PCA,test_PCA]       = kfold(data.pca,data.Y,folds);

% Training Models
disp("TRAINING SVM L1")
[svmL1]                    = svm_main(train,test,folds,epochs,iter,1);
disp("TRAINING SVM L2")
[svmL2]                    = svm_main(train,test,folds,epochs,iter,2);
disp("TRAINING SVM RBF")
[svmRBF]                   = svm_main_k(train,test,folds);
disp("TRAINING SVM RBF")
[svmRBF_pca]               = svm_main_k(train_PCA,test_PCA,folds);
disp("TRAINING SVM L1 PCA")
[svmL1_pca]                = svm_main(train_PCA,test_PCA,folds,epochs,iter,1);
disp("TRAINING SVM L2 PCA")
[svmL2_pca]                =   svm_main(train_PCA,test_PCA,folds,epochs,iter,2);

% Results and plots
C_effect(data,epochs);
L_effect(data,epochs);
ROCcurve(test, svmL1.score, svmL2.score, svmL1_pca.score, svmL2_pca.score,svmRBF.score(:,2),svmRBF_pca.score(:,2));
CM_maker(svmL1, svmL2, svmL1_pca, svmL2_pca ,svmRBF,svmRBF_pca);
Table_maker(svmL1, svmL2, svmL1_pca, svmL2_pca, svmRBF,svmRBF_pca);

%% Preprocess
% Formats data 
function [data] = loadnsort(file)
disp('Loading Data.');
data.raw = load(file);
disp('Data Loaded.');
data.raw = struct2array(data.raw);

% Identifies missing and duplicate data
missingData = sum(sum(ismissing(data.raw)));
[uniqueRows, ~, ~] = unique(data.raw, 'rows', 'stable');
duplicateIndices = setdiff(1:size(data, 1), uniqueRows);

% Removes samples missing data
if missingData==0
    disp('No missing data found.');
else
    disp('Rows missing data found at indices:');
    [rowIndices, colIndices] = find(missingData);
    disp([rowIndices, colIndices]);
    data.raw = rmmissing(data.raw,'MinNumMissing',width(data.raw));
    disp('Rows missing data removed.');
end

% Removes duplicate samples
if isempty(duplicateIndices)
    disp('No duplicates found.');
else
    disp('Duplicate rows found at indices:');
    disp(duplicateIndices);
    data.raw = data.raw(uniqueRows, :);
    disp('Duplicate rows removed.');
end

% Removes outlier 
data.temp = rmoutliers(data.raw,"mean");

% Normalise and creates X dataset 
data.X   = data.temp(:,1:41);
data.Xnorm   = zscore(data.X);

%Creates Y dataset
% Converts 0 to -1 to make classification easier
data.Y   = data.temp(:,42);
data.Y(data.Y == 0) = -1;



end

% Visulises data before and after pre-process
function []=data_vis(data)

% Create a box plot of raw data
figure('Position', [0 0 700 400]);
subplot(2,1,1)
boxplot(data.raw(:,1:41), 'Labels', 1:41);
title('Box Plot of 41 Predictors (Raw Data)');
xlabel('Predictor');
ylabel('Values');
grid on;
% Create a box plot of Preprocessed Data
subplot(2,1,2)
boxplot(data.Xnorm, 'Labels', 1:41);
title('Box Plot of 41 Predictors (Preprocessed Data)');
xlabel('Predictor');
ylabel('Values');
grid on;

end


%% PCA
% Function to perform PCA on input data until a specified variance is reached
function pca_data = pca_data_maker(data, var_target)

var = 0;
n = 1;
% Continue adding principal components until the specified variance is reached
while (var < var_target)
    [pca_data, var, explained] = pcafun(data, n);
    n = n + 1;
end

cumulativeExplainedVar = cumsum(explained);

% plot for showing variance
figure('Position', [0 0 700 300]);
bar(explained,'FaceColor',[(73)/255 (128)/255 (140)/255],'EdgeColor','none'); %73, 128, 140
ylabel('Explained Variance','FontSize', 12);
ylim([0 0.2]);
hold on; 
title('Cumulative Explained Variance','FontSize', 12);
xlabel('Principal Component','FontSize', 12);
yyaxis right
plot(cumulativeExplainedVar, 'k-o');
ylim([0 1]);
ylabel('Cumulative Explained Variance','FontSize', 12);
ax2 = gca;
ax2.YColor = [0 0 0];
grid on;


end
% Function to perform PCA and return reduced data and variance information
function [reduced_X,variance_captured,explained] = pcafun(X,n)
[coeff, ~, ~, explained] = myPCA(X);
% Select the first n principal components
first_n_components = coeff(:, 1:n);
reduced_X = X * first_n_components;

% Calculate the variance captured by the first n components
variance_captured = sum(explained(1:n));
end
% Function to perform PCA and return coefficients, scores, eigenvalues, and explained variance
function [coeff, score, latent, explained] = myPCA(X)
    % Input: X dataset
    % Output: coeff: Principal component coefficients 
    %         score: Principal component scores
    %         latent: Eigenvalues of the covariance matrix
    %         explained: Percentage of total variance explained by each principal component
    
    % covariance matrix
    covMatrix = cov(X);

    % eigenvectors and eigenvalues
    [coeff, latent] = eig(covMatrix);

    % diagonal values 
    latent = diag(latent);

    % descending order
    [latent, idx] = sort(latent, 'descend');
    coeff = coeff(:, idx);

    % Project the data onto the new basis
    score = X * coeff;

    % Normalize eigenvectors 
    coeff = coeff ./ vecnorm(coeff);

    % percentage of total variance 
    explained = latent / sum(latent);
end

%% Data split functions
function [train, test] = kfold(X, Y, folds)
    % Create the indexes for k folds of the dataset
    cv = cvpartition(size(X, 1), 'KFold', folds);
    % Initialises dataset cells
    train.X = cell(folds, 1);
    train.Y = cell(folds, 1);
    test.X = cell(folds, 1);
    test.Y = cell(folds, 1);

    for i = 1:folds
        % Get the training and testing data
        train.X{i} = X(cv.training(i), :);
        train.Y{i} = Y(cv.training(i), :);
        test.X{i} = X(cv.test(i), :);
        test.Y{i} = Y(cv.test(i), :);
    end
end

%% SVM Functions
%Main svm function
function [Result_svm]=svm_main(train,test,folds,epochs,iter,type)

% Data and table initialiseation
varNames = {'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives', 'Accuracy', 'Precision', 'Recall', 'F1Score','C','l'};
varTypes = {'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'};
Result_svm.Vacc = table('Size', [folds, numel(varNames)], 'VariableNames', varNames,'VariableTypes', varTypes);
Result_svm.Tacc = Result_svm.Vacc;
score=[];
acc  =[];

for i=1:folds
    disp(['Step = ' num2str(i) ' of ' num2str(folds)]);
    
    % Optimises Hyperparameters
    [C,l,Baydata(:,(2*i)),Baydata(:,(2*i)-1)] = Bayesian_Optimization_l1(train.X{i},train.Y{i},test.X{i},test.Y{i},epochs,iter,type);
    
    % Selects Type of regulization
    if type == 1
        % L1
        [w, b,acct] = svm_sgd_L1(train.X{i},train.Y{i},test.X{i},test.Y{i}, C, epochs,l);
    elseif type == 2
        % L"
        [w, b,acct] = svm_sgd_L2(train.X{i},train.Y{i},test.X{i},test.Y{i}, C, epochs,l);
    end
    %Stores hyperparameters
    Result_svm.Vacc(i, 9:10) = {C,l};
    Result_svm.Tacc(i, 9:10) = {C,l};
    %Calculates training accuracy and other metrics
    [~,cm1,~] = svm_accuracy(train.X{i},train.Y{i}, w, b);
    [Result_svm.Vacc] = confussion_acc(Result_svm.Vacc,cm1,i);
    %Calculates test accuracy and other metrics
    [~,cm2,scoret] = svm_accuracy(test.X{i},test.Y{i}, w, b);
    [Result_svm.Tacc] = confussion_acc(Result_svm.Tacc,cm2,i);
    % Stores temp values
    score = [score;scoret];
    acc=[acc,acct];

end

% Stores values in to struct
Result_svm.Bayop=Baydata;
Result_svm.score=score;
Result_svm.acc=acc;
disp(" ")
end
% Bayesian Optimization Function
function [Cn,l, EOMT, OMT] = Bayesian_Optimization_l1(X, Y,XT, YT, epochs, iter,type)
    % Hyperparameter definition
    vars = [optimizableVariable('C', [1e-4, 20]),optimizableVariable('l', [1e-4, 20])];
    %Objective function
    fun = @(params) svm_objective(X, Y,XT, YT, params.C,epochs,params.l,type);
    results = bayesopt(fun, vars, 'MaxObjectiveEvaluations', iter,'PlotFcn', {},'Verbose',0);%'PlotFcn', {},'Verbose',0
    %Store hyperparameter
    [Cn,l] = deal(results.XAtMinEstimatedObjective.C,results.XAtMinEstimatedObjective.l);

    EOMT = results.EstimatedObjectiveMinimumTrace;
    OMT = results.ObjectiveMinimumTrace;
end
%Objective function for Bayesian Optimization
function error = svm_objective(X, Y, XT, YT, C,epochs,l,type)
% Selects type of regulization
if type == 1
    [w, b,~] = svm_sgd_L1(X, Y, XT, YT, C, epochs,l);
elseif type == 2
    [w, b,~] = svm_sgd_L2(X, Y, XT, YT, C, epochs,l);
end
    %Calculates current accuracy of model
    [error,~,~] = svm_accuracy(X, Y, w, b);
    error=-error;
end
% SVM L1 Trainer
function [w, b, acc] = svm_sgd_L1(X, Y,XT,YT, C, epochs, lambda)
    % SVM with Stochastic Gradient Descent (SGD) optimization and L1 regularization 
    % Initialize weights and bias
    [m, n] = size(X);
    w = zeros(1, n);
    b = 0;
    lr0 =0.1;
    lr  =lr0;
    % Stochastic Gradient Descent
    for epoch = 1:epochs
        for i = 1:m
            % Select a random data point
            rand_index = randi([1, m]);
            x_i = X(rand_index, :);
            y_i = Y(rand_index);
            
            % Compute hinge loss and its gradient
            loss = 1 - y_i * (w * x_i' + b);
            if loss > 0
                gradient_w = -C * y_i * x_i + lambda * sign(w);  % L1 regularization term with strength lambda
                gradient_b = -C * y_i;
            else
                gradient_w = lambda * sign(w);  % L1 regularization term with strength lambda
                gradient_b = 0;
            end
            
            % Update weights and bias using SGD
            w = w - lr * gradient_w;
            b = b - lr * gradient_b;
        end
        % Update learning rate
        lr = lr0/(1+lambda*epoch);
        acc(epoch,1) = svm_accuracy(XT, YT, w, b);
    end
end
% SVM L2 Trainer
function [w, b,acc] = svm_sgd_L2(X, Y,XT,YT, C, epochs, l)
    % SVM with Stochastic Gradient Descent (SGD) optimization  
    % Initialize weights and bias
    [m, n] = size(X);
    w = zeros(1, n);
    b = 0;
    lr0 = 0.1;
    lr = lr0;
    % Stochastic Gradient Descent
    for epoch = 1:epochs
        for i = 1:m
            % Select a random data point
            rand_index = randi([1, m]);
            x_i = X(rand_index, :);
            y_i = Y(rand_index);
             
            % Compute hinge loss and its gradient
            loss = 1 - y_i * (w * x_i' + b);
            if loss > 0
                gradient_w = -C * y_i * x_i + l * w;  % L2 regularization term
                gradient_b = -C * y_i;
            else
                gradient_w = l * w;  % L2 regularization term
                gradient_b = 0;
            end
            
            % Update weights and bias using SGD
            w = w - lr * gradient_w;
            b = b - lr * gradient_b;
        end
        %Updates learning rate
        lr = lr0/(1+l*epoch);
        acc(epoch,1) = svm_accuracy(XT, YT, w, b);
    end
end
%Accuracy calculator
function [accuracy,cm,score] = svm_accuracy(X, Y, w, b)
    % SVM accuracy calculation
    
    % Parameters:
    %   X: Data matrix
    %   y: True labels
    %   w: Weight vector
    %   b: Bias term
    
    % Predict labels using the trained SVM model
    score = X * w' + b;
    y_pred = sign(score);
    cm = confusionmat(Y, y_pred);
    % Calculate accuracy
    correct_predictions = sum(y_pred == Y);
    total_examples = length(Y);
    accuracy = correct_predictions / total_examples;
end

%% SVM KERNEL
%Main svm function 
function [Result_svm]=svm_main_k(train,test,folds)
% Data and table initialiseation
varNames = {'TruePositives', 'TrueNegatives', 'FalsePositives', 'FalseNegatives', 'Accuracy', 'Precision', 'Recall', 'F1Score'};
varTypes = {'double', 'double', 'double', 'double', 'double', 'double', 'double', 'double'};
Result_svm.Vacc = table('Size', [folds, numel(varNames)], 'VariableNames', varNames,'VariableTypes', varTypes);
Result_svm.Tacc = Result_svm.Vacc;
score=[];

%Runs builtin function for each fold
for i=1:folds
    disp(['SVM L1 reg Step = ' num2str(i) ' of ' num2str(folds)]);

    [cm2,cm1,scoret]=svm_rbf(train.X{i},train.Y{i},test.X{i},test.Y{i}); 
    [Result_svm.Vacc] = confussion_acc(Result_svm.Vacc,cm1,i);
    [Result_svm.Tacc] = confussion_acc(Result_svm.Tacc,cm2,i);

    score = [score;scoret];
end
Result_svm.score=score;
disp(" ")
end

function [CmT,CmV,score]=svm_rbf(X, Y, Xt, Yt)
    % Train SVM model with RBF kernel and automatic parameter tuning
    svm_model = fitcsvm(X, Y, 'KernelFunction', 'RBF', 'OptimizeHyperparameters', 'auto','Verbose',0);
    % Predict labels for the test set
    [y_pred,score] = predict(svm_model, Xt);
    % Create confusion matrix
    CmT = confusionmat(Yt, y_pred);

    % Predict labels for the test set
    [y_pred,~] = predict(svm_model, X);
    % Create confusion matrix
    CmV = confusionmat(Y, y_pred);
end

%% Effect Plots

function C_effect(data,epochs)  
    L=2.5;
    plotdata1=[];
    plotdata2=[];
    plotdata3=[];
    plotdata4=[];

    c = cvpartition(size(data.Xnorm, 1), 'Holdout', 0.3);
    X = data.Xnorm(training(c),:);
    Y = data.Y(training(c),:);
    Xt = data.Xnorm(test(c),:);
    Yt = data.Y(test(c),:);
    fprintf("\nGenerating Graph = ")

    for C = 0:0.01:4
        fprintf("%3.1f%%",(C/4)*100)
        [w, b,~] = svm_sgd_L2(X,Y,Xt,Yt, C, epochs, L);
        [~,cm1,~] = svm_accuracy(X,Y, w, b);
        [~,cm2,~] = svm_accuracy(Xt,Yt, w, b);
        acc1 = (cm1(2, 2) + cm1(1, 1)) / sum(cm1(:));
        acc2 = (cm2(2, 2) + cm2(1, 1)) / sum(cm2(:));
        plotdata1 = [plotdata1;C,acc1];
        plotdata2 = [plotdata2;C,acc2];

        [w, b,~] = svm_sgd_L1(data.Xnorm,data.Y,data.Xnorm,data.Y, C, epochs, L);
        [~,cm1,~] = svm_accuracy(X,Y, w, b);
        [~,cm2,~] = svm_accuracy(Xt,Yt, w, b);
        acc1 = (cm1(2, 2) + cm1(1, 1)) / sum(cm1(:));
        acc2 = (cm2(2, 2) + cm2(1, 1)) / sum(cm2(:));
        plotdata3 = [plotdata3;C,acc1];
        plotdata4 = [plotdata4;C,acc2];
        fprintf(repmat('\b', 1, length(sprintf('%3.1f%%', (C/4)*100))));
    end
    fprintf("DONE \n")
    p1 = polyfit(plotdata1(:,1), plotdata1(:,2), 7);
    p2 = polyfit(plotdata2(:,1), plotdata2(:,2), 7);
    p3 = polyfit(plotdata3(:,1), plotdata3(:,2), 7);
    p4 = polyfit(plotdata4(:,1), plotdata4(:,2), 7);
    ind = 1:5:length(plotdata3);
    x_values = linspace(0, 5, 100);

    figure
    % Plot the first set of data
    plot(plotdata3(ind,1), plotdata3(ind,2), 'o', 'Color', [1 0.8 0.8])
    hold on
    plot(plotdata4(ind,1), plotdata4(ind,2), 'x', 'Color', [1 0.8 0.8])
    plot(x_values, polyval(p3, x_values), 'r--', 'LineWidth', 2); % Trendline for the second set
    plot(x_values, polyval(p4, x_values), 'r-', 'LineWidth', 2); % Trendline for the second set
    
    plot(plotdata1(ind,1), plotdata1(ind,2), 'o', 'Color', [0.7 0.7 1]) 
    plot(plotdata2(ind,1), plotdata2(ind,2), 'x', 'Color', [0.7 0.7 1]) 
    plot(x_values, polyval(p1, x_values), 'b--', 'LineWidth', 2); % Trendline for the first set
    plot(x_values, polyval(p2, x_values), 'b-', 'LineWidth', 2); % Trendline for the first set
    
    axis([0 3 0.4 1]);
    
    legend('L1 Training', 'L1 Test', 'L1 Training Trendline', 'L1 Test Trendline','L2 Training', 'L2 Test','L2 Training Trendline', 'L2 Test Trendline','Location', 'northwest');
    title('Effect of Regularization parameter "C" on model accuracy');
    xlabel('Regularization parameter "C"');
    ylabel('Accuracy of Model');
    hold off
end

function L_effect(data,epochs)  
    C=18;
    plotdata1=[];
    plotdata2=[];
    plotdata3=[];
    plotdata4=[];

    c = cvpartition(size(data.Xnorm, 1), 'Holdout', 0.3);
    X = data.Xnorm(training(c),:);
    Y = data.Y(training(c),:);
    Xt = data.Xnorm(test(c),:);
    Yt = data.Y(test(c),:);
    fprintf("\nGenerating Graph = ")

    for L = 0:0.01:5
        fprintf("%3.1f%%",(L/5)*100)
        [w, b,~] = svm_sgd_L2(X,Y,Xt,Yt, C, epochs, L);
        [~,cm1,~] = svm_accuracy(X,Y, w, b);
        [~,cm2,~] = svm_accuracy(Xt,Yt, w, b);
        acc1 = (cm1(2, 2) + cm1(1, 1)) / sum(cm1(:));
        acc2 = (cm2(2, 2) + cm2(1, 1)) / sum(cm2(:));
        plotdata1 = [plotdata1;L,acc1];
        plotdata2 = [plotdata2;L,acc2];

        [w, b,~] = svm_sgd_L1(data.Xnorm,data.Y,data.Xnorm,data.Y, C, epochs, L);
        [~,cm1,~] = svm_accuracy(X,Y, w, b);
        [~,cm2,~] = svm_accuracy(Xt,Yt, w, b);
        acc1 = (cm1(2, 2) + cm1(1, 1)) / sum(cm1(:));
        acc2 = (cm2(2, 2) + cm2(1, 1)) / sum(cm2(:));
        plotdata3 = [plotdata3;L,acc1];
        plotdata4 = [plotdata4;L,acc2];
        fprintf(repmat('\b', 1, length(sprintf('%3.1f%%', (L/5)*100))));
    end
    fprintf("DONE \n")
    p1 = polyfit(plotdata1(:,1), plotdata1(:,2), 7);
    p2 = polyfit(plotdata2(:,1), plotdata2(:,2), 7);
    p3 = polyfit(plotdata3(:,1), plotdata3(:,2), 7);
    p4 = polyfit(plotdata4(:,1), plotdata4(:,2), 7);
    ind = 1:5:length(plotdata3);
    x_values = linspace(0, 5, 100);

    figure
    % Plot the first set of data
    plot(plotdata3(ind,1), plotdata3(ind,2), 'o', 'Color', [1 0.8 0.8])
    hold on
    plot(plotdata4(ind,1), plotdata4(ind,2), 'x', 'Color', [1 0.8 0.8])
    plot(x_values, polyval(p3, x_values), 'r--', 'LineWidth', 2); % Trendline for the second set
    plot(x_values, polyval(p4, x_values), 'r-', 'LineWidth', 2); % Trendline for the second set
    
    plot(plotdata1(ind,1), plotdata1(ind,2), 'o', 'Color', [0.7 0.7 1]) 
    plot(plotdata2(ind,1), plotdata2(ind,2), 'x', 'Color', [0.7 0.7 1]) 
    plot(x_values, polyval(p1, x_values), 'b--', 'LineWidth', 2); % Trendline for the first set
    plot(x_values, polyval(p2, x_values), 'b-', 'LineWidth', 2); % Trendline for the first set
    
    axis([0 3.5 0.6 1]);
    
    legend('L1 Training', 'L1 Test', 'L1 Training Trendline', 'L1 Test Trendline','L2 Training', 'L2 Test','L2 Training Trendline', 'L2 Test Trendline','Location', 'south west');
    title('Effect of Regularization strength on model accuracy');
    xlabel('Regularization strength');
    ylabel('Accuracy of Model');
    hold off
end

%% Metric
function [result]=confussion_acc(result,confusion_matrix,i)
    % Parameters:
    %   result: Table containing evaluation metrics
    %   confusion_matrix: Confusion matrix
    %   i: Index to store results in the table

    tpos = confusion_matrix(2, 2); % Actual positive and predicted positive
    tneg = confusion_matrix(1, 1); % Actual negative and predicted negative
    fpos = confusion_matrix(1, 2); % Actual negative but predicted positive
    fneg = confusion_matrix(2, 1); % Actual positive but predicted negative
    acc = (tpos + tneg) / sum(confusion_matrix(:));

    % Calculate Precision, Recall, and F1-score
    precision = tpos / (tpos + fpos);
    recall = tpos / (tpos + fneg);
    f1_score = 2 * (precision * recall) / (precision + recall);

    % Create or update a table
    result(i, 1:8) = {tpos, tneg, fpos, fneg, acc, precision, recall, f1_score};

end

function []=ROCcurve(train,S_L1,S_L2,SPCA_L1,SPCA_L2,SRBF,SRBF_PCA)
    
    Y = vertcat(train.Y{:});
    s1 = vertcat(S_L1(:));
    s2 = vertcat(S_L2(:));
    s3 = vertcat(SPCA_L1(:));
    s4 = vertcat(SPCA_L2(:));
    s5 = vertcat(SRBF(:));
    s6 = vertcat(SRBF_PCA(:));
    
    [FPR1, TPR1, ~, AUC1] = perfcurve(Y, s1, 1);
    [FPR2, TPR2, ~, AUC2] = perfcurve(Y, s2, 1);
    [FPR3, TPR3, ~, AUC3] = perfcurve(Y, s3, 1);
    [FPR4, TPR4, ~, AUC4] = perfcurve(Y, s4, 1);
    [FPR5, TPR5, ~, AUC5] = perfcurve(Y, s5, 1);
    [FPR6, TPR6, ~, AUC6] = perfcurve(Y, s6, 1);
    
    % Plot ROC curve
    figure;
    plot(FPR1, TPR1, 'b-', 'LineWidth', 2, 'DisplayName', ['L1       (AUC = ', num2str(AUC1),')']);
    hold on
    plot(FPR3, TPR3, 'b--', 'LineWidth', 2, 'DisplayName',['PCA L1  (AUC = ', num2str(AUC3),')']);
    plot(FPR2, TPR2, 'r-', 'LineWidth', 2, 'DisplayName', ['L2       (AUC = ', num2str(AUC2),')']);
    plot(FPR4, TPR4, 'r--', 'LineWidth', 2, 'DisplayName',['PCA L2  (AUC = ', num2str(AUC4),')']);
    plot(FPR5, TPR5, 'm-', 'LineWidth', 2, 'DisplayName',['RBF  (AUC = ', num2str(AUC5),')']);
    plot(FPR6, TPR6, 'm--', 'LineWidth', 2, 'DisplayName',['PCA RBF  (AUC = ', num2str(AUC6),')']);
    plot([0; 1], [0; 1], 'k-', 'LineWidth', 2,'DisplayName','Random Classifier');
    xlabel('False Positive Rate (1 - Specificity)');
    ylabel('True Positive Rate (Sensitivity)');
    title('ROC Curve');
    legend('show');
    legend('Location', 'southeast'); 
    axis([0 1 0 1]);
end

function CM_maker(svmL1, svmL2, svmL1_pca, svmL2_pca, svmRBF, SRBF_PCA)
    models = {svmL1, svmL1_pca, svmL2, svmL2_pca, svmRBF, SRBF_PCA};
    models_name = {'svm L1', 'svm L1 PCA', 'svm L2', 'svm L2 PCA', 'svm RBF','svm RBF PCA'};

    for i = 1:length(models)
        CM = sum(models{i}.Tacc(:, 1:4));
        CM = table2array(CM);
        CM = [CM(1,1), CM(1,3); CM(1,4), CM(1,2)];

        figure
        confusionchart(CM)
        title(models_name{i})
        
    end
end

function Table_maker(svmL1, svmL2, svmL1_pca, svmL2_pca, svmRBF, SRBF_PCA)
    models = {svmL1, svmL1_pca, svmL2, svmL2_pca, svmRBF,SRBF_PCA};
    models_name = {'svm L1', 'svm L1 PCA', 'svm L2', 'svm L2 PCA', 'svm RBF', 'svm RBF PCA'};
   
    for i = 1:length(models)
        Table1((2*i)-1,:) = table2array(round(mean(models{i}.Vacc(:, 5:8)),3));
        Table1(2*i,:)     = table2array(round(mean(models{i}.Tacc(:, 5:8)),3));
    end

    % Define format
    format0  = '                             | Accuracy | Precision | Recall | F1 score |\n';
    format1  = 'Model : %-30s \n';
    format2  = '        Validation     |   %1.3f    |   %1.3f    | %1.3f  |  %1.3f   |\n';
    format3  = '                 Test     |   %1.3f    |   %1.3f    | %1.3f  |  %1.3f   |\n';
    % Print to the console
    fprintf(format0);
    
    for i = 1:length(models)
        fprintf(format1, models_name{i});
        fprintf(format2, Table1((2*i)-1,:));
        fprintf(format3, Table1((2*i),:));
    end
end

