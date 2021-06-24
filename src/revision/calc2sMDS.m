function mds_coord = calc2sMDS(all_mats, n_dim)    
    %% parse the parameters
    n_tp = size(all_mats, 2);
    n_sub = size(all_mats, 3);
    
    if ~exist('n_dim', 'var')
        n_dim = 10;
    end
    
    
    %% first round of MDS computation
    embed = zeros(n_tp, n_sub, n_dim);
    parfor i_sub = 1 : n_sub
        disp(i_sub/n_sub)
        data_ind = all_mats(:, :, i_sub);      
        D = pdist(data_ind', 'Euclidean');
        Y = mdscale(D, n_dim);
        
        embed(:, i_sub, :) = Y;
    end
    embed = reshape(embed, n_tp, n_sub*n_dim);
    
    %% second round of MDS computation
    D = pdist(embed, 'Euclidean');
    mds_coord = mdscale(D, 3);
end
