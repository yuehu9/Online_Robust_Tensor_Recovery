function L = update_L_col(Reci, lambda1)
[~, col] = size(Reci.L);
A = Reci.A + lambda1 * eye(col);
for j = 1:col
    bj = Reci.B(:,j);
    lj = Reci.L(:,j);
    aj = A(:,j);
    temp = (bj - Reci.L*aj)/A(j,j) + lj;
    L(:,j) = temp/max(norm(temp),1);
end
end