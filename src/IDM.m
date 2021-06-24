function [dm, K, lambda, sigma] = IDM(all_mats, n_dim, n_iter, configAffParams, configDiffParams, disp)
    %% parse the parameters
    n_tp = size(all_mats, 2);
    
    if ~exist('disp', 'var')
        disp = 1;
    end
    
    if ~exist('n_dim', 'var')
        n_dim = 10;
    end
    
    affParams = [];
    if exist('configAffParams','var')
        affParams = setParams(affParams, configAffParams);
    else
        affParams = [];
    end
      
    diffParams.maxInd = n_dim+1;
    if exist('configDiffParams','var')
        diffParams = setParams(diffParams, configDiffParams);
    else
        diffParams = setParams(diffParams, []);
    end
    
    %% iterative diffusion map computation
    lambda = zeros(n_dim, n_iter);
    K = zeros(n_tp, n_tp, n_iter);
    dm = zeros(n_tp, n_dim, n_iter);
    sigma = zeros(n_iter, 1);
    
    data = all_mats;
    for i_iter = 1 : n_iter
        if disp
            fprintf('iter %d\n', i_iter);
        end
        [K(:, :, i_iter), ~, sigma(i_iter)] = calcAffinityMat(data, affParams);
        [dm_temp, lambda, ~, ~, ~, ~] = calcDiffusionMap(K(:, :, i_iter), diffParams);
        dm(:, :, i_iter) = dm_temp(1:n_dim,:)';
        lambda(:, i_iter) = lambda;
        data = dm(:, :, i_iter)';
        
        if disp
            figure(1);
            subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
            scatter3(dm(:, 1, i_iter), dm(:, 2, i_iter), dm(:, 3, i_iter), 'filled');
            title(sprintf('iter %d', i_iter))
            
            figure(2);
            subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
            imagesc(K(:, :, i_iter))
            title(sprintf('iter %d', i_iter))
            
            figure(3);
            subplot(ceil(sqrt(n_iter)), ceil(sqrt(n_iter)), i_iter);
            imagesc(dm(:, :, i_iter))
            title(sprintf('iter %d', i_iter))
        end
    end
    
end
