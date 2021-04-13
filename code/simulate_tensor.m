function [D_all, Sigma_bar_all, X_all, S_all] = simulate_tensor(I1, I2, I3, c, total_n ,ratio_s, ratio_o, magnitude)
% generate sythetic dataset for online alg. 
% Column-wise corrupt along 2nd dimension, each entry drawn random union in [-2,2].
% random masking to creaate missing data
% I1, I2, I3 - dimensions of each tensor minibatch
% c - rank of each minibatch
% total_n - total # of minimatches
% ratio_s -  ratio of sparse corruption
% ratio_o - ratio of observatiion
% D_all - final observation of size (I1, I2, I3*total_n)
% Sigma_bar_all - mask of unboserved data, bool type of size (I1, I2, I3*total_n)
% X_all - ground truth low rank data, corupted entries casted as 0, size (I1, I2, I3*total_n)
% S_all - ground truth of column-wise corruption
% magnitude - magnitude of outliers [-magnitude, magnitude]

% LOW RANK TENSOR: basis
A1 = RandOrthMat(I1, c) ;
A2 = RandOrthMat(I2, c);
A3 = RandOrthMat(I3, c);

D_all = tenzeros(I1, I2, I3*total_n);
Sigma_bar_all = tenzeros(I1, I2, I3*total_n);
X_all = tenzeros(I1, I2, I3*total_n);
S_all = tenzeros(I1, I2, I3*total_n);

       for i = 0:total_n-1
            C = tensor(randn(c,c,c))+1; %core tensor
            X0 = ttm(C,{A1,A2,A3});
            [D, X, S, Sigma_bar, O] = missing_corruptl21(X0, ratio_s, ratio_o, magnitude);
            
            Sigma_bar_all(:, :, i*I3+1:i*I3+I3) = Sigma_bar;
            X_all(:, :, i*I3+1:i*I3+I3) = X;
            S_all(:, :, i*I3+1:i*I3+I3) = S; 
            D_all(:, :, i*I3+1:i*I3+I3) = D;
       end
end

function [D, X, S, Sigma_bar, O] = missing_corruptl21(X, ratioS, ratioO, magnitude)
% corrupt along 2nd (sensor) dim
D1 = size(X,1);
D2 = size(X,2);
D3 = size(X,3);
% sparse tensor, column sparse along 2nd dim
S_ = rand(size(X)) *magnitude*2-magnitude;  %? grossly corrsupted [-2,2]
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
