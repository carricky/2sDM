function [dm, K, embed, lambda1, lambda2, sigma1, sigma2] = calc2sDM(...
    all_mats, n_dim, configAffParams1, configAffParams2, ...
    configDiffParams1, configDiffParams2)
    %Calculates 2-step diffusion maps for 3D matrix
    %
    % [r_pearson, r_rank, y, mask] = calc2sDM(all_mats, dParams_1, dParams_2)
    %
    % Input:      
    % all_mats - time series of every region for all the subjects, by default
    % the second dimension (frame dimension) will be embedded in a lower
    % dimentional space, but the input can be rearranged in dimension to embed
    % other dimentions.  [regions x frames x subjects]
    %
    % n_d - reduced dimension number
    %
    % configAffParams1 - parameters for the first step affinity matrix 
    % calculation, includes fields of 'dist_type', 'kNN', 'self_tune',
    % 'verbose', for more details please refer to utils/calcAffinityMat.m
    % 
    % configAffParams2 - same to configAffParams1, used for the second step 
    % affinity matrix calculation
    %
    % configDiffParams1 - parameters for the first step diffusion maps
    % calculation, includes fields of 'normalization', 't', 'verbose',
    % 'plotResults', for more details please refer to utils/calcDiffusionMap.m
    %
    % configDiffParams2 - same to configDiffParams1, used for the second step 
    % diffusion maps calculation
    %
    % Output:
    % dm - diffusion maps after 2-step diffusion maps reduction, [n_d x frames]
    %
    %
    % Reference: Siyuan Gao, CCN 2018
    % Siyuan Gao, Yale University, 2018-2019
    
    %% parse the parameters
    n_tp = size(all_mats, 2);
    n_sub = size(all_mats, 3);
    
    if ~exist('n_dim', 'var')
        n_dim = 10;
    end
    
    affParams = [];
    if exist('configAffParams1','var')
        affParams1 = setParams(affParams, configAffParams1);
    else
        affParams1 = [];
    end
    
    if exist('configAffParams2','var')
        affParams2 = setParams(affParams, configAffParams2);
    else
        affParams2 = [];
    end
    
    
    diffParams.maxInd = n_dim+1;
    if exist('configDiffParams1','var')
        diffParams1 = setParams(diffParams, configDiffParams1);
    else
        diffParams1 = setParams(diffParams, []);
    end
    
    if exist('configDiffParams2','var')
        diffParams2 = setParams(diffParams, configDiffParams2);
    else
        diffParams2 = setParams(diffParams, []);
    end
    
    
    %% first round of diffusion map computation
    embed = zeros(n_tp, n_sub, n_dim);
    lambda1 = zeros(n_dim, n_sub);
    sigma1 = zeros(n_sub, 1);
    for i_sub = 1 : n_sub
        disp(i_sub/n_sub)
        data_ind = all_mats(:, :, i_sub);      
        [K, ~, sigma1(i_sub)] = calcAffinityMat(data_ind, affParams1);
        [diffusion_map, lambda, ~, ~, ~, ~] = calcDiffusionMap(K, diffParams1);
        % [~,diffusion_map] = pca(data_ind');
        diffusion_map = diffusion_map';
        diffusion_map = diffusion_map ./ sum(lambda);
        embed(:, i_sub, :) = diffusion_map;
        lambda1(:, i_sub) = lambda;
    end
    embed = reshape(embed, n_tp, n_sub*n_dim);
    
    %% second round of diffusion map computation
    [K, ~, sigma2] = calcAffinityMat(embed', affParams2);
    [dm, lambda2, ~, ~, ~, ~] = calcDiffusionMap(K, diffParams2);
end
