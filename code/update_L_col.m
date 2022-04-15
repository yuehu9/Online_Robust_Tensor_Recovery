% function L = update_L_col(Reci, lambda1)
% [~, col] = size(Reci.L);
% A = Reci.A + lambda1 * eye(col);
% for j = 1:col
%     bj = Reci.B(:,j);
%     lj = Reci.L(:,j);
%     aj = A(:,j);
%     temp = (bj - Reci.L*aj)/A(j,j) + lj;
%     L(:,j) = temp/max(norm(temp),1);
% end
% end

function L = update_L_col(Reci_A, Reci_B,  Reci_L, lambda1)
[~, col] = size(Reci_L);
A = Reci_A + lambda1 * eye(col);
L = Reci_L;
for j = 1:col
    bj = Reci_B(:,j);
    lj = Reci_L(:,j);
    aj = A(:,j);
    temp = (bj - Reci_L*aj)/A(j,j) + lj;
    L(:,j) = temp; %/max(norm(temp),1);
end
end


