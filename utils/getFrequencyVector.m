function V_set = getFrequencyVector(train_data, n_dim_f, p_thresh)
    % V_set = getFrequencyVector(train_data, n_dim_f, p_thresh)
    % generate graph Laplacian eigenvectors from graph signal data¡
    n_rg = size(train_data, 1);
    n_sub = size(train_data, 3);
    V_set = zeros(n_rg, n_dim_f, n_sub);
    if ~exist('p_thresh', 'var')
        p_thresh = 0.05;
    end
    
    for i_sub = 1 : n_sub
        fprintf('computing GSP for %dth sub\n', i_sub)
        % GSP decomposition
        x = train_data(:, :, i_sub)';
        % zscore by region
        % TODO: needs to be checked
%         x = zscore(x);
        
        [A, p] = corr(x);
        % thresholding
        A(p > p_thresh) = 0;
        A(A < 0) = 0; % negative connection strength removed to get valid Lap
        L = diag(sum(A, 2)) - A;
        [V, D] = eig(L); % eigendecomposition on L
        [~, order] = sort(diag(D), 'ascend');
        V = V(:, order);
        V_low = V(:, 1:n_dim_f);
        V_set(:, :, i_sub) = V_low;
    end