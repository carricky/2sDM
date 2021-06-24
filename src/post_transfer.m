rng(665)
k = 4;
IDX = kmeans(dm_train(:, :)', k);

% load('/Users/siyuangao/Working_Space/fmri/data/UCLA/rest199.mat')
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% all_signal(missing_nodes, :, :) = [];

% load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_task.mat')
% load('/Users/siyuangao/Working_Space/fmri/data/UCLA/all_label.mat')
% missing_nodes = [249, 239, 243, 129, 266, 109, 115, 118, 250];
% all_task(missing_nodes, :, :) = [];
% test_task = 6;
for ref_sub = 1 : 199
    fprintf('dealing with %dth sub\n', ref_sub)
    % rest

    tar_data = all_signal(:, :, ref_sub);
    tar_data = zscore(tar_data, 0, 1);
    n_tp_test = size(tar_data, 2);
    
    % task
%     tar_data = all_task(:, label==test_task, ref_sub);
%     n_tp_test = sum(label==test_task);
    
    % final config 
    n_tp_total = n_tp_train+n_tp_test;
    test_idx = n_tp_train+1:n_tp_total;
    %% config the parameters
    k = 500;
    configAffParams1.dist_type = 'euclidean';
    configAffParams1.kNN = k;
    configAffParams1.self_tune = 0;
    
    configDiffParams1.t = 1;
    configDiffParams1.normalization='lb';
    
    configAffParams2 = configAffParams1;
    configDiffParams2 = configDiffParams1;
    
    n_dim = 7;   
    %% extend new time points for any subject
    
    % simulate time points for target subject
    % tar_data = zscore(tar_data, 0, 2); % zscore by region
    filter_data = zeros(n_tp_total, n_dim, n_sub);
    for i_sub = 1 : n_sub
        % filter data
        train_data_temp = train_data(:, :, i_sub);
        filter_data(train_idx, :, i_sub) = train_data_temp' * filter_set(:, :, i_sub);
        filter_data(test_idx, :, i_sub) = tar_data' * filter_set(:, :, i_sub);
    end
    filter_data_mat = reshape(filter_data, n_tp_total, n_dim*n_sub);
    filter_data_mat = filter_data_mat';
    [K, ~] = calcAffinityMat(filter_data_mat, configAffParams);
    K = K(n_tp_train+1:n_tp_total, 1:n_tp_train);
    K = K./sum(K, 2);
    psi2 = K*dm_train'./lambda2';
    dm_all = [dm_train, psi2'];
    
    [D_ext, I_ext] = pdist2(dm_all(:, train_idx)', dm_all(:, test_idx)', 'Euclidean', 'Smallest', 10);
    IDX_list = zeros(size(I_ext, 2), 1);
    for i = 1 : size(I_ext, 2)
        IDX_temp = mode(IDX(I_ext(:, i)));
        IDX_list(i) = IDX_temp;
    end
    for i = 1 : 4
        dwell_time_list(ref_sub, i) = (sum(IDX_list==i)/numel(IDX_list));
    end
end

% stat = [mean(dwell_time_list(idx_control,:));
% mean(dwell_time_list(idx_adhd,:));
% mean(dwell_time_list(idx_bpad,:));
% mean(dwell_time_list(idx_schz,:))];
% figure;bar(stat')
% ylim([0,0.6])