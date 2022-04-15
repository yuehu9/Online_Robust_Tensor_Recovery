function [X, E, O, Rec] = OL_rmc21(D, lambda1, lambda2, Rec, Sigma_bar, nrank,outlier_dim, tol, maxIter)
% Feb 2020
% This matlabcode implements the online robust tensor PCA with column-sparse 
% gross corruption.
% D - observation data in tensor format.
% O - recovered non-observed tensor
% Rec - record info. including A, B and L up to now.
% rank - upper bound on rank


addpath tensor_toolbox-master

if nargin < 8
    tol = 1e-7;
elseif tol == -1
    tol = 1e-7;
end

if nargin < 9
    maxIter = 1000;
elseif maxIter == -1
    maxIter = 1000;
end

D_mode = ndims(D);
D_size = size(D);
% tol_size = prod(D_size);                                      

% initialize if it's first round
if isempty(Rec)    
    for i = 1:D_mode
        dim_i = D_size(i);
        Rec{i}.A = zeros(nrank, nrank);
        Rec{i}.B = zeros(dim_i, nrank);
        
        Rec{i}.L = rand(dim_i, nrank);
    end
end

% update R, E, O
[R, E, O] = solve_proj_21(D, Rec,nrank, lambda1, lambda2, Sigma_bar,outlier_dim, tol, maxIter);

% calculate X
% Xhat = cell(D_mode,1);
% for i = 1:D_mode
%     Dmat = tenmat(D, i);
%     Dmat(:, :) = Rec{i}.L * R{i}';
%     Xhat{i} = tensor(Dmat);
% end
% X = tenfun(@sum,Xhat{:})./ D_mode;

X = zeros(size(D));
for i = 1:D_mode
    Dmat = tenmat(D, i);
    Dmat(:, :) = Rec{i}.L * R{i}';
    X = X + tensor(Dmat);
end
X = X./ D_mode;

% update L
for i = 1:D_mode
    Rec{i}.A = Rec{i}.A + R{i}.' * R{i};
    Rec{i}.B = Rec{i}.B + double(tenmat(D - E, i)) * R{i};
    Rec{i}.L = update_L_col(Rec{i}.A, Rec{i}.B, Rec{i}.L, lambda1);
end



end