function [R, E, O, iter] = solve_proj_21(D, Rec, nrank, lambda1, lambda2, Sigma_bar, outlier_dim, tol, maxIter)
%
%     solve the problem:
%     min_{R, E} sum_i(0.5*|D-Li*Ri-E-O|_F^2 + 0.5*lambda1* sum_i(|Ri|_F^2) + lambda2*|E|_2,1
%       s.t. O_(Sigma) = 0
%     
%      
%
    converged = false;
    iter = 0;

    % initiatiate
    D_mode = ndims(D);
    D_size = size(D);
    tol_size = prod(D_size);
    for i = 1:D_mode
        dim_c = tol_size / D_size(i);
    %     disp([num2str(dim_c) num2str(nrank)])
        R{i} = zeros(dim_c, nrank);
    end
    I = eye(nrank);
    O = tenzeros( size(D));
    E = tenzeros( size(D));

    while ~converged
        iter = iter+1;
        stopCriterion = 0;

        % update E
        Res = tenzeros(size(D));
        for i = 1: D_mode
            % record residual (Bi - Li*Ri - O) for E
            Res_i = tenmat(D - O, i);
    %         disp([num2str(size(L)) num2str(size(R{i}'))])
            Res_i(:,:) = double(Res_i) - Rec{i}.L * R{i}';
            Res = Res + tensor(Res_i);
        end
        E_old = E;
        temp_E = Res./D_mode;
        temp_Em = tenmat(temp_E,outlier_dim);
        for j = 1:size(temp_Em,2)
            temp_Em(:,j) = temp_Em(:,j) * max(0,1-lambda2/(D_mode*norm(temp_Em(:,j))));
        end 
        E = tensor(temp_Em); 
        % cal. convergence criteria
        stopCriterion = max(stopCriterion, norm(E - E_old)/norm(D));

        % update R
        R_old = R;
        for i = 1: D_mode
            L = Rec{i}.L;
            RT = (L' * L + lambda1 * I)\(L' * double(tenmat(D - E - O,i)) );
            R{i} = RT';
            % cal. convergence criteria
            stopCriterion = max(stopCriterion, norm(R{i} - R_old{i})/norm(D));
        end

        % update O
        ResO = tenzeros(size(D));
        for i = 1: D_mode
            % record residual (Bi - Li*Ri - E) for O
            ResO_i = tenmat(D - E, i);
    %         disp([num2str(size(L)) num2str(size(R{i}'))])
            ResO_i(:,:) = double(ResO_i) - Rec{i}.L * R{i}';
            ResO = ResO + tensor(ResO_i);
        end
        O_temp = ResO./D_mode;
        O = Sigma_bar .* O_temp;

        if stopCriterion < tol
            converged = true;
        end    

    %     if mod( total_svd, 10) == 0
    %         disp(['#svd ' num2str(total_svd) ' r(A) ' num2str(rank(X_hat))...
    %             ' |E|_0 ' num2str(length(find(abs(E_hat)>0)))...
    %             ' stopCriterion ' num2str(stopCriterion)]);
    %     end    
    %     
    
    
        if ~converged && iter >= maxIter
%             disp('Maximum iterations reached') ;
            converged = 1 ;    
        end
        
    end

end