% This code tests if it can recover manually corrupts noaa data

%% load data
addpath tensor_toolbox-master
addpath ..
addpath PROPACK
clear variables;
load('../data/Noaa_12M.mat')
rng('default');
rng(150);

%% construct observation matrix into tensor fromat
nl = size(Obs2,1);        % #links
nm = 24 ;         % #hours in a day
nd = size(Obs2,2)/nm;     % #days

outlier_dim = 2; 
ratio_s = 0.05; % ratio of sparse corruption
ratio_o = 0.9; % ratio of observatiion

epoch = 1; % online training repeat epochs
% % flip the second epoch
% Obs2_flip = flip(Obs2, 2);
% D0 = [Obs2, Obs2_flip, Obs2];

D0 = tensor(Obs2,[nl nm nd*epoch]);

Xhat_OL = tenzeros(nl, nm, nd*epoch); % record onlie recovery
Shat_OL = tenzeros(nl, nm, nd*epoch); % record recovery
% pollute
[D_all, X_all, S_all, Sigma_bar_all, O_all] = missing_corruptl21(D0, ratio_s, ratio_o);

%% online
n_day = size(D0, 3);

dimension = nl;
nrank = 20;
lambda1 = 0.01; % optimazation parameter
lambda2 = 1/sqrt(log(dimension*dimension))*70;

Rec = [];
cur_iter = 1;

total_time = 0;

for i = 1:n_day
    % days as minibatch
    D = D_all(:, :, i );
    X = X_all(:, :, i );
    S = S_all(:, :, i);
    Sigma_bar = Sigma_bar_all(:, :,i );
    
    D = squeeze(D);
    X = squeeze(X);
    S = squeeze(S);    

    Rec_old = Rec;
    tic
    [Xhat, Shat, Ohat, Rec] = OLRTR(D, lambda1, lambda2, Rec, Sigma_bar, nrank,outlier_dim, 1e-3, 50);
    run_time = toc;
    total_time = total_time + run_time;
    Xhat_OL(:, :,i) = Xhat;
    Shat_OL(:, :,i) = Shat;
    
    cur_iter = cur_iter+1;
end  
disp(['online totoal run time', num2str(total_time)])


% total loss
disp([newline 'performance for all samples '])
thresh = 10; 
% online
[res, f1, precision, recall] = cal_rmse_f1(Xhat_OL, X_all, Shat_OL, S_all, outlier_dim, thresh);
disp(['online, low rank re: ' num2str(res) '; f1: ' num2str(f1) ])
disp(['precision: ' num2str(precision) '; recall: ' num2str(recall) ])

st = 10;
disp(['discard first ' num2str(st)  ' samples '])
% online
[res, f1, precision, recall] = cal_rmse_f1(Xhat_OL(:,:,st+1:end), X_all(:,:,st+1:end), ...
        Shat_OL(:,:,st+1:end), S_all(:,:,st+1:end), outlier_dim, thresh);
disp(['online, low rank re: ' num2str(res) '; f1: ' num2str(f1) ])



%% funciton

function [D, X, S, Sigma_bar, O] = missing_corruptl21(X, ratioS, ratioO)
% corrupt along 2nd (sensor) dim
D1 = size(X,1);
D2 = size(X,2);
D3 = size(X,3);
% sparse tensor, column sparse along 2nd dim
S_ = rand(size(X))*20 + 30;  %? grossly corrsupted
S_ = tensor(S_, size(X));
% S_  = reshape(S_, [14,24,1]);

Col_ = rand(D1,D3);
Col_S = Col_<=ratioS;
Col_S = Col_<= ratioS; % percentage of corrupted column
Col_S = repmat(Col_S, [1 1 D2]);
Col_S = tensor(Col_S);
Col_S = permute(Col_S,[1 3 2]);
S = S_ .* Col_S;


% Corresponding column for X is 0
All = tenones(size(X));
Col_X = All - Col_S;
X = X .* Col_X;

% Observation matrix
D_ = S + X;

% partial observation
Sigma_ = rand(size(X));
Sigma = Sigma_<= ratioO; % keep data by ratio
Sigma_bar = tensor(Sigma_ > ratioO); % index of missing entry
Sigma = tensor(Sigma, size(X));

S = S.* Sigma;
D = D_ .* Sigma;
O = D_ - D;
end

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
    
    %find out outlier cols; only compare unpolluted columns of X?
    Shat_m = tenmat(Shat,outlier_dim); 
    S_m =  tenmat(S,outlier_dim);
    col_Shat = any(abs(double(Shat_m)) > thresh);  %find index all nonzero coloumns of E
    col_S = any(abs(double(S_m)) > thresh);
    [precision, recall, f1] = cal_f1(col_S,col_Shat);
    
    X_m = tenmat(Xhat,outlier_dim);
    X_m(:,col_S) = 0;
    res = norm(double(X_m) - double(tenmat(X,outlier_dim))) / norm(double(tenmat(X,outlier_dim)));
end
