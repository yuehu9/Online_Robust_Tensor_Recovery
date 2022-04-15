% This code tests if it can recover manually corrupts noaa data

%% load data
addpath tensor_toolbox-master
addpath ..
addpath PROPACK
clear variables;

type = 'DEW'; % Dew pint
% type = 'MA1';  % atmospheric pressure
% type = 'TEM'; % temperature
filename = ['E:\onlineRPCA\data\noaa_' type '.mat'];
load(filename)

rng('default');
rng(15);

% fill tiny bit missing data
[Obs2,~] = fillmissing(Obs2,'linear');

%% construct observation matrix into tensor fromat
nl = size(Obs2,1);        % #links
nm = 24 ;         % #hours in a day
nd = size(Obs2,2)/nm;     % #days

outlier_dim = 2; 
ratio_s = 0.05; % ratio of sparse corruption
ratio_o = 0.9; % ratio of observatiion

epoch = 1; % online training repeat epochs

D0 = tensor(Obs2,[nl nm nd*epoch]);

Xhat_OL = tenzeros(nl, nm, nd*epoch); % record onlie recovery
Shat_OL = tenzeros(nl, nm, nd*epoch); % record recovery
% pollute
[D_all, X_all, S_all, Sigma_bar_all, O_all] = missing_corruptl21(D0, ratio_s, ratio_o);

%% online
n_day = size(D0, 3);
loss_rec = zeros(n_day,1);
f1_rec = zeros(n_day,3);
dimension = nl;

if strcmp(type, 'TEM') == 1
    nrank = 3;
    alpha = 330;
    lambda1 = 0.01;
elseif strcmp(type, 'DEW') == 1
    nrank = 3;
    alpha = 420;
    lambda1 = 0.01;
elseif strcmp(type, 'MA1') == 1
    nrank = 5;
    alpha = 500;
    lambda1 = 0.01;
end

lambda2 = 1/sqrt(log(dimension*dimension))*alpha;
    
Rec = [];
cur_iter = 1;
L = [];



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
    [Xhat, Shat, Ohat, Rec] = OLRTR(D, lambda1, lambda2, Rec, Sigma_bar, nrank,outlier_dim, 1e-3, 10);

    Xhat_OL(:, :,i) = Xhat;
    Shat_OL(:, :,i) = Shat;
   
    cur_iter = cur_iter+1;
end  



%% total loss
disp([ 'performance for all samples '])
if strcmp(type, 'TEM') == 1
     thresh = 0.01; 
elseif strcmp(type, 'DEW') == 1
   thresh = 0.01; 
elseif strcmp(type, 'MA1') == 1
    thresh = 2; 
end
    
    
% online
[res, f1, precision, recall] = cal_rmse_f1(Xhat_OL, X_all, Shat_OL, S_all, outlier_dim, thresh);
disp(['obs ratio ' num2str(ratio_o)]);
disp(['online, low rank re: ' num2str(res) '; f1: ' num2str(f1) ])
disp(['precision: ' num2str(precision) '; recall: ' num2str(recall) ])

disp(newline)




%% funciton
function R = row_corrcoef(mat1, mat2)
nl = size(mat1, 1);
R = zeros(nl,1);
for i = 1: nl  
    index = ~isnan(mat1(i,:)) & ~isnan(mat2(i,:));
    coef = corrcoef(mat1(i,index), mat2(i,index));
    R(i) = coef(2,1);
end
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

function X_m = remove_anomaly(Xhat, S, outlier_dim, thresh)
    S_m =  tenmat(S,outlier_dim);
    col_S = any(abs(double(S_m)) > thresh);
    X_m = tenmat(Xhat,outlier_dim);
    X_m(:,col_S) = 0;
    X_m = tensor(X_m);
end