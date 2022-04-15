% This code tests if it can recover manually corrupts noaa data

%% load data
% addpath tensor_toolbox-master
addpath tensor_toolbox-v321
addpath ..
addpath PROPACK
clear variables;
load('E:\onlineRPCA\Data\aot_12M.mat') 
load('../Data/Noaa_chi_12M.mat');
rng('default');
rng(150);

%% fill missing data
[Obs2_fill,~] = fillmissing(Obs2,'linear');
%% construct observation matrix into tensor fromat
nl = size(Obs2,1);        % #links
nm = 24 ;         % #hours in a day
nd = size(Obs2,2)/nm;     % #days

outlier_dim = 2; 
epoch = 3; % online training repeat epochs
% flip the second epochc
Obs2_flip = flip(Obs2, 2);
D_all = [Obs2, Obs2_flip, Obs2];

Sigma_bar_all = isnan(D_all);
Sigma_bar_all = tensor(Sigma_bar_all,[nl nm nd*epoch]);

D_all(isnan(D_all)) = 0;
D_all = tensor(D_all,[nl nm nd*epoch]);


Xhat_OL = tenzeros(nl, nm, nd*epoch); % record online recovery
Shat_OL = tenzeros(nl, nm, nd*epoch); % record recovery

%cal missing rate
Size = numel(double(Obs2));
missing_rate = sum(sum(isnan(Obs2))) / Size;

%% online

dimension = nl;
nrank = 3;
lambda1 = 0.01; % optimazation parameter
lambda2 = 1/sqrt(log(dimension*dimension))*370;

Rec = [];
cur_iter = 1;
rng(15);
total_time = 0;

for i = 1:nd*epoch
    % days as minibatch
    D = D_all(:, :, i );
    Sigma_bar = Sigma_bar_all(:, :,i );
    
    D = squeeze(D);

    tic
    [Xhat, Shat, Ohat, Rec] = OL_rmc21(D, lambda1, lambda2, Rec, Sigma_bar, nrank,outlier_dim, 1e-3, 50);
    run_time = toc;
    total_time = total_time + run_time;
    Xhat_OL(:, :,i) = Xhat;
    Shat_OL(:, :,i) = Shat;
    
    cur_iter = cur_iter+1;
end  
disp(['online totoal run time', num2str(total_time)])


%% examing results
% pearson corr
Xhat_mat = double(tenmat(Xhat_OL,1))';

Xhat_mat = Xhat_mat(end - length(noaa)+1:end ,:);

Xhat_A = [Xhat_mat, noaa'];
R_recover = corrcoef(Xhat_A,'Rows','complete');
avg_R_recover = mean(R_recover(end, 1:end-1));

Obs2_0 = Obs2;
Obs2_0(isnan(Obs2_0)) = 0;
X_A = [Obs2_0',noaa' ];
R_raw = corrcoef(X_A,'Rows','complete');
avg_R_raw = mean(R_raw(end, 1:end-1));

disp(['Correlation, recovered: ', num2str(avg_R_recover) ' original: ', num2str(avg_R_raw)])
disp(['recovered range: '])
sprintf('[%.0f, %.0f]', min(min(Xhat_mat)), max(max(Xhat_mat)))

