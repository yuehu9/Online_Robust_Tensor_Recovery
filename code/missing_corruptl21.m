% corrupt given noaa data with corruption ratio ratioS, entries pof gross
% corruption drawn from Uniform distribution [min_u. max_u]; observation
% ratio ratioO
function [D, X, S, Sigma_bar, O] = missing_corruptl21(X, ratioS, ratioO, min_u, max_u)
    if nargin < 4  % default uniform in range larger than 10% of the original
        min_u = min(min( tenmat(X, 1).data)) ;
        min_u = min_u - abs(min_u) * 0.4;
        max_u = max(max( tenmat(X, 1).data)) ;
        max_u = max_u + abs(max_u) * 0.4;
    end
% corrupt along 2nd (sensor) dim
D1 = size(X,1);
D2 = size(X,2);
D3 = size(X,3);
% sparse tensor, column sparse along 2nd dim
S_ = rand(size(X))*(max_u-min_u) + min_u;  %? grossly corrsupted
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

% partial observationn
Sigma_ = rand(size(X));
Sigma = Sigma_<= ratioO; % keep data by ratio
Sigma_bar = tensor(Sigma_ > ratioO); % index of missing entry
Sigma = tensor(Sigma, size(X));

S = S.* Sigma;
D = D_ .* Sigma;
O = D_ - D;
end