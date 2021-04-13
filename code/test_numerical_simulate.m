% experiment of OLRTR on numerically simulate seies of low rank tensor with fiber-wise corruption
% and missing data
%% 
clear;
addpath tensor_toolbox-master ;
addpath PROPACK;
rng('default');
rng(6);

ratio_s = 0.05; % ratio of sparse corruption
ratio_o = 0.9; % ratio of observatiion
% tensor dimension of each mini-batch (I,I,I)
I1 = 50; 
I2 = 50;
I3 = 50;
dimension = I1;

c = 3;  % tucker rank of low rank tensor (c,c,c)
outlier_dim = 2;
total_n = 100; %total number of mini-batches


magnitude = 2; % magnitude of outliers
[D_all, Sigma_bar_all, X_all, S_all] = simulate_tensor(I1, I2, I3, c, total_n ,ratio_s, ratio_o,magnitude);


 %% RPCA
% record onlie recovery
Xhat_OL = tenzeros(I1, I2, I3*total_n); 
Shat_OL = tenzeros(I1, I2, I3*total_n);

Rec = [];
nrank = 3; % target rank

% record of RE and F1 score for each sample
loss_rec = zeros(total_n,1);
f1_rec = zeros(total_n,3); 

verbal = true;
cur_iter = 1;
total_time = 0;
for i = 0:total_n-1
    D = D_all(:, :, i*I3+1:i*I3+I3);
    Sigma_bar = Sigma_bar_all(:, :, i*I3+1:i*I3+I3);
    X = X_all(:, :, i*I3+1:i*I3+I3);
    S = S_all(:, :, i*I3+1:i*I3+I3);
    % OLRTR
    lambda1 = 0.01; % optimazation parameter
    lambda2 = 1/sqrt(log(dimension*dimension))*3;
    tic;
    [Xhat, Shat, Ohat, Rec] = OLRTR(D, lambda1, lambda2, Rec, Sigma_bar, nrank,outlier_dim, 1e-4, 500);
    run_time = toc;
    total_time = total_time + run_time;
    Xhat_OL(:, :, i*I3+1:i*I3+I3) = Xhat;
    Shat_OL(:, :, i*I3+1:i*I3+I3) = Shat;
    
    thresh = 1;
    [res, f1, precision, recall] = cal_rmse_f1(Xhat, X, Shat, S, outlier_dim, thresh);
    
    if verbal
        disp([newline num2str(cur_iter) 'th sample, time: ' num2str(run_time)  ', re: ' num2str(res)])
        loss_rec(cur_iter) = res;

        S_m =  tenmat(S,outlier_dim);
        col_S = any(abs(double(S_m)) > thresh);
        % precision/recall 
        if sum(col_S)>0
            disp(['sparse precision: ' num2str(precision) '; recall: ' num2str(recall) '; F1: ' num2str(f1)])
            f1_rec(cur_iter,:) = [precision recall f1];
        else
            f1_rec(cur_iter,:) = [NaN NaN NaN];
        end
    end
    cur_iter = cur_iter+1;
end

disp(['online totoal run time: ', num2str(total_time)])


%% plot loss
figure()
subplot(2,1,1)
plot(loss_rec)
ylim([0,1])
title('RE')
 
subplot(2,1,2)
xx = 1:total_n;
scatter(xx',f1_rec(:,3))
ylim([0,1])
title('F1 score')
 
%% loss
thresh = 1;
% online
[res, f1, precision, recall] = cal_rmse_f1(Xhat_OL, X_all, Shat_OL, S_all, outlier_dim, thresh);
disp('all samples ')
disp(['low rank re: ' num2str(res) '; f1: ' num2str(f1) ])
disp(['precision: ' num2str(precision) '; recall: ' num2str(recall) ])

n_samples = total_n*0.9*I3;
disp([newline 'last ' num2str(n_samples/I3) ' samples '])
% online
[res, f1, precision, recall] = cal_rmse_f1(Xhat_OL(:,:,end-n_samples :end), X_all(:,:,end-n_samples :end), ...
        Shat_OL(:,:,end-n_samples :end), S_all(:,:,end-n_samples :end), outlier_dim, thresh);
disp(['low rank re: ' num2str(res) '; f1: ' num2str(f1) ])


%% function 
function [precision, recall, f1] = cal_f1(col_S,col_Shat )
    tp = sum(sum(sum(sum((col_S==1) & (col_Shat==1)))));
    fn = sum(sum(sum(sum((col_S==1) & (col_Shat==0)))));
    fp = sum(sum(sum(sum((col_S==0) & (col_Shat==1)))));
    precision = tp/(tp+fp);
    recall = tp/(tp+fn);
    f1 = 2 * (precision * recall) / (precision + recall);
end

function [res, f1, precision, recall] = cal_rmse_f1(Xhat, X, Shat, S, outlier_dim, thresh)
    if nargin < 6
        thresh = 5;
    end
    
    %find out outlier cols; only compare unpolluted columns of X
    Shat_m = tenmat(Shat,outlier_dim); 
    S_m =  tenmat(S,outlier_dim);
    col_Shat = any(abs(double(Shat_m)) > thresh);  %find index all nonzero coloumns of E
    col_S = any(abs(double(S_m)) > thresh);
    [precision, recall, f1] = cal_f1(col_S,col_Shat);
    
    X_m = tenmat(Xhat,outlier_dim);
    X_m(:,col_S) = 0;
    res = norm(double(X_m) - double(tenmat(X,outlier_dim))) / norm(double(tenmat(X,outlier_dim)));
end